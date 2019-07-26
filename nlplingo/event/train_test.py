from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging

import numpy as np

from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.embeddings.word_embeddings import WordEmbeddingFactory
from nlplingo.event.argument.run import generate_argument_data_feature
from nlplingo.event.argument.run import test_argument
from nlplingo.event.argument.run import train_argument
from nlplingo.event.event_domain import EventDomain

from nlplingo.event.trigger.run import generate_trigger_data_feature
from nlplingo.event.trigger.run import get_predicted_positive_triggers
from nlplingo.event.trigger.run import test_trigger
from nlplingo.event.trigger.run import train_trigger_from_file

from nlplingo.nn.extractor import Extractor

logger = logging.getLogger(__name__)


def decode_trigger_argument(params, word_embeddings, trigger_extractors, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractors: list[nlplingo.nn.extractor.Extractor] # trigger extractors
    :type argument_extractors: list[nlplingo.nnl.extractor.Extractor] # argument extractors
    """
    trigger_extractor = None
    if len(trigger_extractors) > 1:
        raise RuntimeError('More than one trigger model cannot be used in decoding.')
    elif len(trigger_extractors) == 1:
        trigger_extractor = trigger_extractors[0]

    if len(argument_extractors) == 0:
        raise RuntimeError('At least one argument extractor must be specified to decode over arguments.')

    if trigger_extractor is None:
        raise RuntimeError('Trigger extractor must be specified in parameter file.')

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    logging.info('#### Generating trigger examples')
    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_extractor.example_generator, test_docs, trigger_extractor.feature_generator)

    predicted_positive_triggers = []
    if len(trigger_examples) > 0:
        logging.info('#### Predicting triggers')
        trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)
        predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples,
                                                                  trigger_extractor.domain)

    predictions_output_file = params['predictions_file']
    clusters = {}

    for docid in predicted_positive_triggers:
        for t in predicted_positive_triggers[docid]:
            """:type: nlplingo.event.trigger.example.EventTriggerExample"""
            logging.info('PREDICTED-ANCHOR {} {} {} {} {}'.format(t.sentence.docid, t.event_type, '%.4f' % t.score, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
            cluster = clusters.setdefault(t.event_type, dict())
            sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))), dict())
            sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
            sentence['eventType'] = t.event_type
            sentence['score'] = '%.4f' % (t.score)
            sentence['docId'] = t.sentence.docid
            sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
            trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
            trigger_array = trigger.setdefault('trigger', list())
            trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
            trigger_array = sorted(list(set(trigger_array)))
            trigger['trigger'] = trigger_array


    actor_ner_types = set(['PER', 'ORG', 'GPE'])
    place_ner_types = set(['GPE', 'FAC', 'LOC', 'ORG'])
    time_ner_types = set(['TIMEX2.TIME'])

    if len(predicted_positive_triggers) > 0:
        for argument_extractor in argument_extractors:
            logging.info('Generating argument examples based on predicted triggers')
            (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                generate_argument_data_feature(
                    argument_extractor.example_generator, test_docs, argument_extractor.feature_generator,
                    predicted_triggers=predicted_positive_triggers)

            if len(arg_examples_pt) == 0:
                continue

            logging.info('Predicting arguments based on predicted triggers')
            argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)
            pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

            for i, predicted_label in enumerate(pred_arg_max):
                if predicted_label != extractor.domain.get_event_role_index('None'):
                    eg = arg_examples_pt[i]
                    """:type: nlplingo.event.argument.example.EventArgumentExample"""
                    eg.score = argument_predictions_pt[i][predicted_label]
                    predicted_role = extractor.domain.get_event_role_from_index(predicted_label)

                    if predicted_role == 'Time' and eg.argument.label not in time_ner_types:
                        continue
                    if predicted_role == 'Place' and eg.argument.label not in place_ner_types:
                        continue
                    if predicted_role == 'Actor' and eg.argument.label not in actor_ner_types:
                        continue

                    print('PREDICTED-ARGUMENT {} {} {} {} {}'.format(eg.sentence.docid, predicted_role, '%.4f' % (eg.score), eg.argument.start_char_offset(), eg.argument.end_char_offset()))
                    cluster = clusters.setdefault(eg.anchor.label, dict())
                    sentence = cluster.setdefault(str((str(eg.sentence.docid), str(eg.sentence.int_pair.to_string()))), dict())
                    if sentence.get('token', None) is None:
                        print("Something is wrong")
                        sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(eg.sentence.tokens)]
                    trigger = sentence.setdefault('trigger_{}'.format(eg.anchor.int_pair.to_string()), dict())
                    argument = trigger.setdefault(predicted_role, list())
                    # argument.extend([tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens])
                    argument_array = [tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]
                    argument.append((min(argument_array), max(argument_array)))
                    argument = sorted(list(set(argument)))
                    trigger[predicted_role] = argument

    with open(predictions_output_file, 'w') as fp:
            json.dump(clusters, fp, indent=4, sort_keys=True)


def load_embeddings(params):
    """
    :return: dict[str : WordEmbeddingAbstract]
    """
    embeddings = dict()

    if 'embeddings' in params:
        embeddings_params = params['embeddings']
        word_embeddings = WordEmbeddingFactory.createWordEmbedding(
            embeddings_params.get('type', 'word_embeddings'),
            embeddings_params
        )
        embeddings['word_embeddings'] = word_embeddings
        print('Word embeddings loaded')

    if 'dependency_embeddings' in params:
        dep_embeddings_params = params['dependency_embeddings']
        dependency_embeddings = WordEmbeddingFactory.createWordEmbedding(
            dep_embeddings_params.get('type', 'dependency_embeddings'),
            dep_embeddings_params
        )
        embeddings['dependency_embeddings'] = dependency_embeddings
        print('Dependency embeddings loaded')

    return embeddings


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)

    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    # ==== loading of embeddings ====
    embeddings = load_embeddings(params)

    load_extractor_models_from_file = False
    if args.mode in {'test_trigger', 'test_argument', 'decode_trigger_argument', 'decode_trigger'}:
        load_extractor_models_from_file = True

    trigger_extractors = []
    argument_extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractor = Extractor(params, extractor_params, embeddings, load_extractor_models_from_file)
        if extractor.model_type.startswith('event-trigger_'):
            trigger_extractors.append(extractor)
        elif extractor.model_type.startswith('event-argument_'):
            argument_extractors.append(extractor)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(extractor.model_type))

    if 'domain_ontology.scoring' in params:
        scoring_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology.scoring'), 'scoring')
    else:
        scoring_domain = None

    if args.mode == 'train_trigger_from_file':
        train_trigger_from_file(params, embeddings, trigger_extractors[0])
    elif args.mode == 'test_trigger':
        test_trigger(params, embeddings, trigger_extractors[0])
    elif args.mode == 'train_argument':
        train_argument(params, embeddings, argument_extractors[0])
    elif args.mode == 'test_argument':
        test_argument(params, embeddings, trigger_extractors[0], argument_extractors[0], scoring_domain)
    elif args.mode == 'decode_trigger_argument':
        decode_trigger_argument(params, embeddings, trigger_extractors, argument_extractors)
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))
