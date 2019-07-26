from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import logging

from collections import defaultdict

import numpy as np

from keras.models import load_model as keras_load_model

from nlplingo.common.scoring import evaluate_f1
from nlplingo.common.scoring import print_score_breakdown
from nlplingo.common.scoring import write_score_to_file

from nlplingo.annotation.ingestion import prepare_docs
from nlplingo.event.trigger.generator import EventTriggerExampleGenerator


from nlplingo.event.trigger.generator import EventKeywordList
from nlplingo.event.trigger.metric import get_recall_misses as get_trigger_recall_misses
from nlplingo.event.trigger.metric import get_precision_misses as get_trigger_precision_misses

logger = logging.getLogger(__name__)

def get_predicted_positive_triggers(predictions, examples, event_domain):
    """Collect the predicted positive triggers and organize them by docid
    Also, use the predicted event_type for each such trigger example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    none_class_index = event_domain.get_event_type_index('None')
    assert len(predictions ) ==len(examples)
    ret = defaultdict(list)

    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            eg = examples[i]
            """:type: nlplingo.event.trigger.trigger_example.EventTriggerExample"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            eg.anchor.label = eg.event_type
            eg.score = predictions[i][index]
            ret[eg.sentence.docid].append(eg)
    return ret

def generate_trigger_data_feature(example_generator, docs, feature_generator):
    """
    :type example_generator: nlplingo.event.trigger.generator.EventTriggerExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator
    """


    examples = example_generator.generate(docs, feature_generator)
    """:type: list[nlplingo.event.trigger.example.EventTriggerExample]"""

    data = EventTriggerExampleGenerator.examples_to_data_dict(examples, feature_generator.features)

    data_list = [np.asarray(data[k]) for k in feature_generator.features.feature_strings]

    label = np.asarray(data['label'])

    print('#trigger-examples=%d' % (len(examples)))
    return (examples, data, data_list, label)


def train_trigger_from_file(params, word_embeddings, trigger_extractor):
    """
    :type params: dict
    :type word_embeddings: dict[str:nlplingo.embeddings.word_embeddings.WordEmbedding]
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    """

    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)
    test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings)

    # only retain the event triggers and argument that are in ontology
    for doc in train_docs:
        doc.apply_domain(trigger_extractor.domain)
    for doc in test_docs:
        doc.apply_domain(trigger_extractor.domain)

    feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator"""

    example_generator = trigger_extractor.example_generator
    """:type: nlplingo.event.trigger.generator.EventTriggerExampleGenerator"""

    trigger_model = trigger_extractor.extraction_model
    """:type: nlplingo.nn.trigger_model.TriggerModel"""

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(trigger_model)={}'.format(type(trigger_model)))

    (train_examples, train_data, train_data_list, train_label) = generate_trigger_data_feature(
        example_generator, train_docs, feature_generator)
    print(train_label)

    (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(
        example_generator, test_docs, feature_generator)

    trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    # TODO: review the following later
    if trigger_extractor.hyper_parameters.finetune_epoch > 0:
        for layer in trigger_model.model.layers:
            #if layer.name in (u'sentence_embedding',
            #                  u'lex_embedding',
            #                  u'word_input_left',
            #                  u'word_input_right'
            #              ):
            layer.trainable = True
        trigger_model.model.compile(optimizer=trigger_model.optimizer, loss=u'categorical_crossentropy', metrics=[])
        trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    predictions = trigger_model.predict(test_data_list)

    if params['save_model']:
        print('==== Saving Trigger model ====')
        trigger_model.save_keras_model(trigger_extractor.model_file)


    predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples, trigger_extractor.domain)
    # { docid -> list[nlplingo.event.trigger.trigger_example.EventTriggerExample] }

    single_word_prediction = 0
    multi_word_prediction = 0
    for docid in predicted_positive_triggers:
        for eg in predicted_positive_triggers[docid]:
            length = len(eg.anchor.tokens)
            if length == 1:
                single_word_prediction += 1
            elif length > 1:
                multi_word_prediction += 1
    print('** #single_word_prediction={}, #multi_word_prediction={}'.format(single_word_prediction, multi_word_prediction))

    score, score_breakdown = evaluate_f1(predictions, test_label, trigger_extractor.domain.get_event_type_index('None'))
    logging.info(score.to_string())

    print_score_breakdown(trigger_extractor, score_breakdown)

    write_score_to_file(trigger_extractor, score, score_breakdown, params['train.score_file'])

    if 'test' in params['data']:
        test_trigger(params, word_embeddings, trigger_extractor)


def train_trigger_from_feature(params, extractor):
    train_data = np.load(params['data']['train']['features'])
    train_data_list = train_data['data_list']
    train_label = train_data['label']

    test_data = np.load(params['data']['dev']['features'])
    test_data_list = test_data['data_list']
    test_label = test_data['label']

    trigger_model = extractor.extraction_model
    """:type: nlplingo.model.trigger_model.TriggerModel"""

    trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    print('==== Saving Trigger model ====')
    trigger_model.save_keras_model(extractor.model_file)


def load_trigger_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'trigger.hdf'))
    return model
    #with open(os.path.join(model_dir, 'trigger.pickle'), u'rb') as f:
    #    trigger_model = pickle.load(f)
    #    """:type: nlplingo.model.event_cnn.EventExtractionModel"""
        #trigger_model.load_keras_model(filename=os.path.join(model_dir, 'trigger.hdf'))
    #    trigger_model.model_dir = model_dir
    #    return trigger_model


def load_trigger_modelfile(filepath):
    return keras_load_model(str(filepath))


def test_trigger(params, word_embeddings, trigger_extractor):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    """

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    for doc in test_docs:
        doc.apply_domain(trigger_extractor.domain)

    feature_generator = trigger_extractor.feature_generator
    """:type: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator"""

    example_generator = trigger_extractor.example_generator
    """:type: nlplingo.event.trigger.generator.EventTriggerExampleGenerator"""

    trigger_model = trigger_extractor.extraction_model
    """:type: nlplingo.nn.trigger_model.TriggerModel"""

    (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(
        example_generator, test_docs, feature_generator)

    predictions = trigger_model.predict(test_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples, trigger_extractor.domain)

    score, score_breakdown = evaluate_f1(predictions, test_label, trigger_extractor.domain.get_event_type_index('None'))
    logging.info(score.to_string())

    label_arg_max = np.argmax(test_label, axis=1)
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, v in enumerate(label_arg_max):
        g_label = trigger_extractor.domain.get_event_type_from_index(label_arg_max[i])
        p_label = trigger_extractor.domain.get_event_type_from_index(pred_arg_max[i])
        if g_label != 'None' and p_label != g_label:
            logging.debug('RECALL-ERROR: {} {} {} {}'.format(
                test_examples[i].anchor.text.encode('ascii', 'ignore'),
                test_examples[i].sentence.text.encode('ascii', 'ignore'),
                g_label,
                p_label
            )
            )

    for i, v in enumerate(label_arg_max):
        g_label = trigger_extractor.domain.get_event_type_from_index(label_arg_max[i])
        p_label = trigger_extractor.domain.get_event_type_from_index(pred_arg_max[i])
        if p_label != 'None' and p_label != g_label:
            logging.debug('PRECISION-ERROR: {} {} {} {}'.format(
                test_examples[i].anchor.text.encode('ascii', 'ignore'),
                test_examples[i].sentence.text.encode('ascii', 'ignore'),
                g_label,
                p_label
            )
            )

    print_score_breakdown(trigger_extractor, score_breakdown)
    write_score_to_file(trigger_extractor, score, score_breakdown, params['test.score_file'])

    return predicted_positive_triggers


def test_trigger_list(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    generator = EventTriggerExampleGenerator(event_domain, params)
    event_keyword_list = EventKeywordList(params.get_string('event_keywords'))
    event_keyword_list.print_statistics()

    test_docs = prepare_docs(params.get_string('filelist.test'), word_embeddings)

    for doc in test_docs:
        doc.apply_domain(event_domain)
    # constraint_event_type_to_domain(test_docs, event_domain)

    examples = generator.generate(test_docs)
    test_label = np.asarray([eg.label for eg in examples])

    predictions = []
    for eg in examples:
        token = eg.anchor.tokens[0]

        event_types = event_keyword_list.get_event_types_for_tokens([token])

        event_type = 'None'
        if len(event_types) == 1:
            event_type = list(event_types)[0]
        event_index = event_domain.get_event_type_index(event_type)

        eg_predictions = np.zeros(len(event_domain.event_types), dtype=params.get_string('cnn.int_type'))
        eg_predictions[event_index] = 1
        predictions.append(eg_predictions)

    number_of_recall_miss = 0
    recall_misses = get_trigger_recall_misses(predictions, test_label, event_domain.get_event_type_index('None'),
                                              event_domain, examples)
    for key in sorted(recall_misses.keys()):
        count = recall_misses[key]
        if count > 0:
            print('Trigger-recall-miss\t{}\t{}'.format(key, count))
            number_of_recall_miss += count
    print('Total# of recall miss={}'.format(number_of_recall_miss))

    number_of_precision_miss = 0
    precision_misses = get_trigger_precision_misses(predictions, test_label, event_domain.get_event_type_index('None'),
                                                    event_domain, examples)
    for key in sorted(precision_misses.keys()):
        count = precision_misses[key]
        if count > 0:
            print('Trigger-precision-miss\t{}\t{}'.format(key, count))
            number_of_precision_miss += count
    print('Total# of precision miss={}'.format(number_of_precision_miss))

    score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'))
    print(score.to_string())

    for index, f1_score in score_breakdown.items():
        et = event_domain.get_event_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

    # label_arg_max = np.argmax(test_label, axis=1)
    # pred_arg_max = np.argmax(predictions, axis=1)
    # print(label_arg_max)
    # for i, v in enumerate(label_arg_max):
    #     g_label = event_domain.get_event_type_from_index(label_arg_max[i])
    #     p_label = event_domain.get_event_type_from_index(pred_arg_max[i])
    #     if g_label != 'None' and p_label != g_label:
    #         print('RECALL-ERROR: {} {}'.format(test_examples[i].anchor.text.encode('ascii', 'ignore'), test_examples[i].sentence.text.encode('ascii', 'ignore')))

    output_dir = params.get_string('output_dir')
    with open(os.path.join(output_dir, 'test_trigger.score'), 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            et = event_domain.get_event_type_from_index(index)
            f.write('{}\t{}\n'.format(et, f1_score.to_string()))


def decode_trigger(params, word_embeddings, extractor):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type extractor: nlplingo.model.Extractor
    """
    trigger_generator = extractor.generator

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_generator,
        test_docs,
        extractor.model_type,
        extractor.model_flags
    )

    predictions_output_file = params['predictions_file']
    clusters = {}

    print('==== Loading Trigger model ====')
    trigger_model = load_trigger_modelfile(extractor.model_file)
    trigger_predictions = trigger_model.predict(trigger_data_list)

    predicted_positive_triggers = get_predicted_positive_triggers(
        trigger_predictions,
        trigger_examples,
        extractor.domain.get_event_type_index('None'),
        extractor.domain
    )

    for docid in predicted_positive_triggers:
        for t in predicted_positive_triggers[docid]:
            """:type: nlplingo.event.event_trigger.EventTriggerExample"""
            print('PREDICTED-ANCHOR {} {} {} {}'.format(t.sentence.docid, t.event_type, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
            cluster = clusters.setdefault(t.event_type, dict())
            sentence = cluster.setdefault(str((str(t.sentence.docid), str(t.sentence.int_pair.to_string()))), dict())
            sentence['token'] = [str((idx, token.text)) for idx, token in enumerate(t.sentence.tokens)]
            sentence['eventType'] = t.event_type
            sentence['docId'] = t.sentence.docid
            sentence['sentenceOffset'] = (t.sentence.int_pair.first, t.sentence.int_pair.second)
            trigger = sentence.setdefault('trigger_{}'.format(t.anchor.int_pair.to_string()), dict())
            trigger_array = trigger.setdefault('trigger', list())
            trigger_array.append((t.anchor.tokens[0].index_in_sentence, t.anchor.tokens[-1].index_in_sentence))
            trigger_array = sorted(list(set(trigger_array)))
            trigger['trigger'] = trigger_array

    with open(predictions_output_file, 'w') as fp:
            json.dump(clusters, fp, indent=4, sort_keys=True)

