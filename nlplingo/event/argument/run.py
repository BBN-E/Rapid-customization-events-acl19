from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import numpy as np

from keras.models import load_model as keras_load_model

from nlplingo.common.scoring import evaluate_arg_f1
from nlplingo.annotation.ingestion import prepare_docs

from nlplingo.event.argument.generator import EventArgumentExampleGenerator

from nlplingo.event.trigger.run import generate_trigger_data_feature
from nlplingo.event.trigger.run import get_predicted_positive_triggers

logger = logging.getLogger(__name__)


def load_argument_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'argument.hdf'))
    return model

def load_argument_modelfile(filepath):
    return keras_load_model(str(filepath))



def generate_argument_data_feature(generator, docs, feature_generator, predicted_triggers=None):
    """
    +1
    :type generator: nlplingo.event.argument.generator.EventArgumentExampleGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type feature_generator: nlplingo.event.argument.feature.EventArgumentFeatureGenerator
    :type predicted_triggers: defaultdict(list[nlplingo.event.trigger.example.EventTriggerExample])
    """
    examples = generator.generate(docs, feature_generator, triggers=predicted_triggers)
    """:type: list[nlplingo.event.argument.example.EventArgumentExample]"""

    data = EventArgumentExampleGenerator.examples_to_data_dict(examples, feature_generator.features)
    data_list = [np.asarray(data[k]) for k in feature_generator.features.feature_strings]

    label = np.asarray(data['label'])

    print('#arg-examples=%d' % (len(examples)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (examples, data, data_list, label)


def argument_modeling(params, train_docs, test_docs, event_domain, word_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    #:type predicted_positive_triggers: list[nlplingo.event.event_trigger.EventTriggerExample]
    """

def train_argument(params, word_embeddings, argument_extractor):
    """
    :type params: dict
    :type word_embeddings: dict[str:nlplingo.embeddings.word_embeddings.WordEmbedding]
    :type argument_extractor: nlplingo.nn.extraactor.Extractor
    """
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)
    test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings)

    for doc in train_docs:
        doc.apply_domain(argument_extractor.domain)
    for doc in test_docs:
        doc.apply_domain(argument_extractor.domain)

    feature_generator = argument_extractor.feature_generator
    example_generator = argument_extractor.example_generator
    argument_model = argument_extractor.extraction_model

    logger.debug('type(feature_generator)={}'.format(type(feature_generator)))
    logger.debug('type(example_generator)={}'.format(type(example_generator)))
    logger.debug('type(argument_model)={}'.format(type(argument_model)))

    (train_examples, train_data, train_data_list, train_label) = generate_argument_data_feature(
        example_generator,
        train_docs,
        feature_generator
    )
    print(train_label)

    (test_examples, test_data, test_data_list, test_label) = generate_argument_data_feature(
        example_generator,
        test_docs,
        feature_generator
    )

    argument_model.fit(train_data_list, train_label, test_data_list, test_label)

    predictions = argument_model.predict(test_data_list)

    score, score_breakdown, gold_labels = evaluate_arg_f1(argument_extractor.domain, test_label, test_examples, predictions)
    print('Arg-score: ' + score.to_string())

    for index, f1_score in score_breakdown.items():
        er = argument_extractor.domain.get_event_role_from_index(index)
        print('{}\t{}'.format(er, f1_score.to_string()))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            er = argument_extractor.domain.get_event_role_from_index(index)
            f.write('{}\t{}\n'.format(er, f1_score.to_string()))

    print('==== Saving Argument model ====')
    if params['save_model']:
        argument_model.save_keras_model(argument_extractor.model_file)


def test_argument(params, word_embeddings, trigger_extractor, argument_extractor, scoring_domain=None):
    """
    :type params: dict
    :type trigger_extractor: nlplingo.nn.extractor.Extractor
    :type argument_extractor: nlplingo.nn.extractor.Extractor
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type scoring_domain: nlplingo.event.event_domain.EventDomain
    """
    event_domain = argument_extractor.domain

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)
    for doc in test_docs:
        doc.apply_domain(event_domain)

    logging.info('Generating trigger examples')
    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_extractor.example_generator, test_docs, trigger_extractor.feature_generator)

    logging.info('Generating argument examples based on gold triggers')
    (arg_examples, arg_data, arg_data_list, arg_label) = generate_argument_data_feature(
        argument_extractor.example_generator, test_docs, argument_extractor.feature_generator)

    for v in arg_data_list:
        print(v)

    logging.info('Predicting triggers')
    trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples, trigger_extractor.domain)

    logging.info('Predicting arguments based on gold triggers')
    argument_predictions = argument_extractor.extraction_model.predict(arg_data_list)

    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_arg_f1(
        event_domain, arg_label, arg_examples, argument_predictions, scoring_domain)
    print('Arg-scores with gold-triggers: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with gold-triggers: {}\t{}'.format(er, f1_score.to_string()))

    logging.info('Generating argument examples based on predicted triggers')
    (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
        generate_argument_data_feature(
            argument_extractor.example_generator, test_docs, argument_extractor.feature_generator,
            predicted_triggers=predicted_positive_triggers)

    logging.info('Predicting arguments based on predicted triggers')
    argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)

    # evaluate arguments with predicted triggers
    score_with_predicted_triggers, score_breakdown_with_predicted_triggers, pred_labels = \
        evaluate_arg_f1(event_domain, arg_label_pt, arg_examples_pt, argument_predictions_pt, scoring_domain, gold_labels=gold_labels)

    print('Arg-scores with predicted-triggers: {}'.format(score_with_predicted_triggers.to_string()))

    for index, f1_score in score_breakdown_with_predicted_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with predicted-triggers: {}\t{}'.format(er, f1_score.to_string()))

    output_file = params['test.score_file']
    with open(output_file, 'w') as f:
        f.write('With-gold-triggers: {}\n'.format(score_with_gold_triggers.to_string()))
        for index, f1_score in score_breakdown_with_gold_triggers.items():
            er = event_domain.get_event_role_from_index(index)
            f.write('With gold-triggers: {}\t{}\n'.format(er, f1_score.to_string()))
        f.write('With-predicted-triggers: {}\n'.format(score_with_predicted_triggers.to_string()))
        for index, f1_score in score_breakdown_with_predicted_triggers.items():
            er = event_domain.get_event_role_from_index(index)
            f.write('With predicted-triggers: {}\t{}\n'.format(er, f1_score.to_string()))

