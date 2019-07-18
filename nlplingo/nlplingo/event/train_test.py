from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import codecs
import re
import sys
import json

import spacy

import pickle

from collections import defaultdict

import numpy as np

from keras.models import load_model as keras_load_model




from nlplingo.text.text_span import EventArgument

from nlplingo.annotation.ingestion import read_doc_annotation

from nlplingo.embeddings.word_embeddings import WordEmbeddingFactory
from nlplingo.event.event_trigger import EventTriggerGenerator
from nlplingo.event.event_trigger import EventKeywordList
from nlplingo.event.event_trigger import get_recall_misses as get_trigger_recall_misses
from nlplingo.event.event_trigger import get_precision_misses as get_trigger_precision_misses
from nlplingo.event.event_argument import EventArgumentGenerator
from nlplingo.event.argument import ArgumentGenerator

from nlplingo.event.event_domain import EventDomain
from nlplingo.event.event_domain import CyberDomain
from nlplingo.event.event_domain import AceDomain
from nlplingo.event.event_domain import PrecursorDomain
from nlplingo.event.event_domain import UIDomain
from nlplingo.event.event_domain import AcePrecursorDomain

import nlplingo.common.io_utils as io_utils
from nlplingo.common.parameters import Parameters
from nlplingo.event.event_sentence import EventSentenceGenerator
from nlplingo.event.event_mention import EventMentionGenerator
from nlplingo.event.event_pair import EventPairGenerator
from nlplingo.event.event_pair import EventPairData
from nlplingo.event.event_pair import print_pair_predictions
from nlplingo.event.novel_event_type import NovelEventType

from nlplingo.model.extractor import Extractor
from nlplingo.model.sentence_model import MaxPoolEmbeddedSentenceModel
from nlplingo.model.pair_model import MaxPoolEmbeddedPairModel

from nlplingo.common.scoring import evaluate_f1
from nlplingo.common.scoring import evaluate_arg_f1
from nlplingo.common.scoring import  evaluate_accuracy
from nlplingo.common.scoring import evaluate_f1_binary


def prepare_docs(filelists, word_embeddings):
    # read IDT and ENote annotations
    docs = read_doc_annotation(io_utils.read_file_to_list(filelists))
    print('num# docs = %d' % (len(docs)))

    # apply Spacy for sentence segmentation and tokenization, using Spacy tokens to back Anchor and EntityMention
    spacy_en = spacy.load('en')
    for index, doc in enumerate(docs):
        if len(doc.sentences) == 0:
            doc.annotate_sentences(word_embeddings, spacy_en)
        else:
            # read_doc_annotation above has already done sentence splitting and tokenization to construct
            # Sentence objects, e.g. using output from CoreNLP
            doc.annotate_sentences(word_embeddings, model=None)

        if (index % 20) == 0:
            print('Prepared {} input documents out of {}'.format(str(index+1), str(len(docs))))

    # em_labels = set()
    # for doc in docs:
    #     for sentence in doc.sentences:
    #         for em in sentence.entity_mentions:
    #             em_labels.add(em.label)
    # for label in sorted(em_labels):
    #     print('EM-LABEL {}'.format(label))
    # print(len(em_labels))

    number_anchors = 0
    number_args = 0
    number_assigned_anchors = 0
    number_assigned_args = 0
    number_assigned_multiword_anchors = 0
    event_type_count = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            number_anchors += event.number_of_anchors()
            number_args += event.number_of_arguments()
            event_type_count[event.label] += 1
        for sent in doc.sentences:
            for event in sent.events:
                number_assigned_anchors += event.number_of_anchors()
                number_assigned_args += event.number_of_arguments()
                for anchor in event.anchors:
                    if len(anchor.tokens) > 1:
                        number_assigned_multiword_anchors += 1

    # print('In train_test.prepare_docs')
    # for doc in docs:
    #     for sent in doc.sentences:
    #         for event in sent.events:
    #             for arg in event.arguments:
    #                 if len(arg.entity_mention.tokens) > 1:
    #                     print('Multiword argument: {} {}'.format(arg.entity_mention.label, ' '.join(token.text for token in arg.entity_mention.tokens)))
    # exit(0)

    print('In %d documents, #anchors=%d #assigned_anchors=%d #assigned_multiword_anchors=%d, #args=%d #assigned_args=%d' % \
          (len(docs), number_anchors, number_assigned_anchors, number_assigned_multiword_anchors, number_args, number_assigned_args))
    #print('Event type counts:')
    #for et in sorted(event_type_count.keys()):
    #    print('#{}: {}'.format(et, event_type_count[et]))
    return docs

def get_predicted_positive_triggers(predictions, examples, none_class_index, event_domain):
    """Collect the predicted positive triggers and organize them by docid
    Also, use the predicted event_type for each such trigger example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    assert len(predictions)==len(examples)
    ret = defaultdict(list)
    
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            #print(predictions[i][index])
            eg = examples[i]
            """:type: nlplingo.event.event_trigger.EventTriggerExample"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            eg.anchor.label = eg.event_type
            eg.score = predictions[i][index]
            ret[eg.sentence.docid].append(eg)
    return ret


def generate_trigger_data_feature(generator, docs, extractor_name, model_flags):
    """
    :type generator: nlplingo.event.event_trigger.EventTriggerGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    :type extractor_name: str
    :type model_flags: dict
    """
    examples = generator.generate(docs)
    data = generator.examples_to_data_dict(examples)

    if extractor_name == 'event-trigger_cnn':
        data_list = [
            np.asarray(data['word_vec']),
        ]
        if model_flags.get('use_lex_info'):
            data_list.append(np.asarray(data['lex']))

        data_list.append(np.asarray(data['pos_array']))

        if model_flags.get('use_bio_index', False):
            data_list.append(np.asarray(data['entity_type_array']))

    elif extractor_name == 'event-trigger_piecewise-cnn':
        data_list = [
            np.asarray(data['word_vec_left']),
            np.asarray(data['word_vec_right']),
            np.asarray(data['pos_array_left']),
            np.asarray(data['pos_array_right']),
            np.asarray(data['lex'])
        ]
        if model_flags.get('use_bio_index', False):
            data_list.extend(
                [
                    np.asarray(data['entity_type_array_left']),
                    np.asarray(data['entity_type_array_right'])
                ]
            )
    elif extractor_name == 'event-trigger_lstm':
        data_list = [
            np.asarray(data['word_vec_forward']),
            np.asarray(data['word_vec_backward'])
        ]
    elif extractor_name == 'event-trigger_lstmcnn':
        data_list = [
            np.asarray(data['word_vec']),
            np.asarray(data['word_vec_forward']),
            np.asarray(data['word_vec_backward']),
            np.asarray(data['lex']),
            np.asarray(data['pos_array'])
        ]

        if model_flags.get('use_bio_index', False):
            data_list.append(np.asarray(data['entity_type_array']))
    elif extractor_name == 'event-trigger_embedded':
        data_list = [
            np.asarray(data['lex'])
        ]
    elif extractor_name == 'event-trigger_onlineembedded':
        data_list = [
            np.asarray(data['vector_data_array'])
        ]

    else:
        raise RuntimeError('Extractor name {} not implemented'.format(extractor_name))

    label = np.asarray(data['label'])

    print('#trigger-examples=%d' % (len(examples)))
    # print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    #     len(data['word_vec']), len(data['pos_array']), len(data['label'])))
    return (examples, data, data_list, label)

# def trigger_modeling(params, train_docs, test_docs, extractor):
#     """
#     :type params: dict
#     :type train_docs: list[nlplingo.text.text_theory.Document]
#     :type test_docs: list[nlplingo.text.text_theory.Document]
#     :type extractor: nlplingo.model.extractor.Extractor
#     """
#     generator = extractor.generator
#     """:type: nlplingo.event.event_trigger.EventTriggerGenerator"""
#
#     use_bio_index = extractor.model_flags['use_bio_index']
#     (train_examples, train_data, train_data_list, train_label) = generate_trigger_data_feature(generator, train_docs, extractor.name, use_bio_index)
#     np.savez(params['data']['train']['features'], data_list=np.asarray(train_data_list), label=train_label)
#
#     (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(generator, test_docs, extractor.name, use_bio_index)
#     np.savez(params['data']['dev']['features'], data_list=np.asarray(test_data_list), label=test_label)
#
#     trigger_model = extractor.extraction_model
#     """:type: nlplingo.model.trigger_model.TriggerModel"""
#
#     trigger_model.fit(train_data_list, train_label, test_data_list, test_label)
#
#     print('==== Saving Trigger model ====')
#     trigger_model.save_keras_model(extractor.model_file)
#     # with open(os.path.join(output_dir, 'trigger.pickle'), u'wb') as f:
#     #    pickle.dump(trigger_model, f)
#
#     if len(test_examples) == 0:
#         return []
#
#     predictions = trigger_model.predict(test_data_list)
#     predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples, extractor.domain.get_event_type_index('None'), extractor.domain)
#
#     single_word_prediction = 0
#     multi_word_prediction = 0
#     for docid in predicted_positive_triggers:
#         for eg in predicted_positive_triggers[docid]:
#             length = len(eg.anchor.tokens)
#             if length == 1:
#                 single_word_prediction += 1
#             elif length > 1:
#                 multi_word_prediction += 1
#     print('** single_word_prediction={}, multi_word_prediction={}'.format(single_word_prediction, multi_word_prediction))
#
#     # calculate the recall denominator
#     number_test_anchors = 0
#     for doc in test_docs:
#         for event in doc.events:
#             number_test_anchors += event.number_of_anchors()
#
#     score, score_breakdown = evaluate_f1(predictions, test_label, extractor.domain.get_event_type_index('None'), num_true=number_test_anchors)
#     print(score.to_string())
#
#     for index, f1_score in score_breakdown.items():
#         et = extractor.domain.get_event_type_from_index(index)
#         print('{}\t{}'.format(et, f1_score.to_string()))
#
#     with open(os.path.join(params['output_dir'], 'train_trigger.score'), 'w') as f:
#         f.write(score.to_string() + '\n')
#         for index, f1_score in score_breakdown.items():
#             et = extractor.domain.get_event_type_from_index(index)
#             f.write('{}\t{}\n'.format(et, f1_score.to_string()))
#
#     return predicted_positive_triggers

def print_event_statistics(docs):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    stats = defaultdict(int)
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                et = event.label
                stats[et] += 1
                for arg in event.arguments:
                    if arg.label == 'CyberAttackType' and et == 'CyberAttack':
                        print('CyberAttackType {} [{}] {}'.format(event.id, arg.text, sent.text.encode('ascii', 'ignore')))
                for anchor in event.anchors:
                    stats['{}.{}'.format(et, 'anchor')] += 1
                    if et == 'CyberAttack':
                        print('CyberAttack <{}> {}'.format(anchor.text, sent.text.encode('ascii', 'ignore')))
                for arg in event.arguments:
                    role = '{}.{}'.format(et, arg.label)
                    stats[role] += 1
    for key in sorted(stats.keys()):
        print('{}\t{}'.format(key, str(stats[key])))

# def constraint_event_type_to_domain(docs, event_domain):
#     """
#     :type docs: list[nlplingo.text.text_theory.Document]
#     :type event_domain: nlplingo.event.event_domain.EventDomain
#     """
#     for doc in docs:
#         doc.events = [event for event in doc.events if event.label in event_domain.event_types.keys()]
#     for doc in docs:
#         for sent in doc.sentences:
#             sent.events = [event for event in sent.events if event.label in event_domain.event_types.keys()]

def train_trigger_from_file(params, word_embeddings, extractor, from_model=False):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type extractor: nlplingo.model.extractor.Extractor
    """
    print('==== Preparing training docs ====')
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)
    print('==== Preparing dev docs ====')
    test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings)

    for doc in train_docs:
        doc.apply_domain(extractor.domain)
    for doc in test_docs:
        doc.apply_domain(extractor.domain)

    generator = extractor.generator
    """:type: nlplingo.event.event_trigger.EventTriggerGenerator"""

    (train_examples, train_data, train_data_list, train_label) = generate_trigger_data_feature(
        generator,
        train_docs,
        extractor.model_type,
        extractor.model_flags
    )
    #np.savez(params['data']['train']['features'], data_list=train_data_list[0], label=train_label)

    (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(
        generator,
        test_docs,
        extractor.model_type,
        extractor.model_flags
    )
    #np.savez(params['data']['dev']['features'], data_list=np.asarray(test_data_list), label=test_label)

    trigger_model = extractor.extraction_model
    """:type: nlplingo.model.trigger_model.TriggerModel"""

    trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    if extractor.hyper_parameters['fine-tune_epoch'] > 0:
        # TODO: should I simply be setting all layers to be trainable?
        for layer in trigger_model.model.layers:
            if layer.name in (u'sentence_embedding',
                              u'lex_embedding',
                              u'word_input_left',
                              u'word_input_right'
                          ):
                layer.trainable = True
        trigger_model.model.compile(optimizer=trigger_model.optimizer, loss=u'categorical_crossentropy', metrics=[])
        trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    if params['save_model']:
        print('==== Saving Trigger model ====')
        trigger_model.save_keras_model(str(extractor.model_file))

    predictions = trigger_model.predict(test_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples, extractor.domain.get_event_type_index('None'), extractor.domain)
    # { docid -> list[nlplingo.event.event_trigger.EventTriggerExample] }

    # TODO: modularize the following into a scoring method, and check the code interns

    single_word_prediction = 0
    multi_word_prediction = 0
    for docid in predicted_positive_triggers:
        for eg in predicted_positive_triggers[docid]:
            length = len(eg.anchor.tokens)
            if length == 1:
                single_word_prediction += 1
            elif length > 1:
                multi_word_prediction += 1
    print('** single_word_prediction={}, multi_word_prediction={}'.format(single_word_prediction, multi_word_prediction))

    # calculate the recall denominator
    number_test_anchors = 0
    for doc in test_docs:
        for event in doc.events:
            number_test_anchors += event.number_of_anchors()


    #score, score_breakdown = evaluate_f1(predictions, test_label, extractor.domain.get_event_type_index('None'), num_true=number_test_anchors)
    score, score_breakdown = evaluate_f1(predictions, test_label, extractor.domain.get_event_type_index('None'))
    print(score.to_string())

    for index, f1_score in score_breakdown.items():
        et = extractor.domain.get_event_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            et = extractor.domain.get_event_type_from_index(index)
            f.write('{}\t{}\n'.format(et, f1_score.to_string()))

    if 'test' in params['data']:
        test_trigger(params, word_embeddings, extractor, trigger_model=trigger_model)


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

def load_argument_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'argument.hdf'))
    return model
    #with open(os.path.join(model_dir, 'argument.pickle'), u'rb') as f:
    #    argument_model = pickle.load(f)
    #    """:type: nlplingo.model.event_cnn.EventExtractionModel"""
    #    #argument_model.load_keras_model(filename=os.path.join(model_dir, 'argument.hdf'))
    #    argument_model.model_dir = model_dir
    #    return argument_model

def load_argument_modelfile(filepath):
    return keras_load_model(str(filepath))
def load_trigger_modelfile(filepath):
    return keras_load_model(str(filepath))


def test_trigger(params, word_embeddings, extractor):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type extractor: nlplingo.model.extractor.Extractor
    """
    generator = extractor.generator
    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    for doc in test_docs:
        doc.apply_domain(extractor.domain)


    (test_examples, test_data, test_data_list, test_label) = generate_trigger_data_feature(
        generator,
        test_docs,
        extractor.model_type,
        extractor.model_flags
    )

    print('==== Loading Trigger model ====')
    trigger_model = extractor.extraction_model

    predictions = trigger_model.predict(test_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples,
                                                                  extractor.domain.get_event_type_index('None'),
                                                                  extractor.domain)

    score, score_breakdown = evaluate_f1(predictions, test_label, extractor.domain.get_event_type_index('None'))
    print(score.to_string())

    for index, f1_score in score_breakdown.items():
        et = extractor.domain.get_event_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))

    label_arg_max = np.argmax(test_label, axis=1)
    pred_arg_max = np.argmax(predictions, axis=1)
    print(label_arg_max)
    for i, v in enumerate(label_arg_max):
        g_label = extractor.domain.get_event_type_from_index(label_arg_max[i])
        p_label = extractor.domain.get_event_type_from_index(pred_arg_max[i])
        if g_label != 'None' and p_label != g_label:
            print('RECALL-ERROR: {} {} {} {}'.format(
                    test_examples[i].anchor.text.encode('ascii', 'ignore'), 
                    test_examples[i].sentence.text.encode('ascii', 'ignore'), 
                    g_label, 
                    p_label
                    )
                  )
   
    for i, v in enumerate(label_arg_max):
        g_label = extractor.domain.get_event_type_from_index(label_arg_max[i])
        p_label = extractor.domain.get_event_type_from_index(pred_arg_max[i])
        if p_label != 'None' and p_label != g_label:
            print('PRECISION-ERROR: {} {} {} {}'.format(
                    test_examples[i].anchor.text.encode('ascii', 'ignore'), 
                    test_examples[i].sentence.text.encode('ascii', 'ignore'), 
                    g_label, 
                    p_label
                    )
                  )

    with open(params['test.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            et = extractor.domain.get_event_type_from_index(index)
            f.write('{}\t{}\n'.format(et, f1_score.to_string()))

    return predicted_positive_triggers


def test_trigger_list(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    generator = EventTriggerGenerator(event_domain, params)
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
    recall_misses = get_trigger_recall_misses(predictions, test_label, event_domain.get_event_type_index('None'), event_domain, examples)
    for key in sorted(recall_misses.keys()):
        count = recall_misses[key]
        if count > 0:
            print('Trigger-recall-miss\t{}\t{}'.format(key, count))
            number_of_recall_miss += count
    print('Total# of recall miss={}'.format(number_of_recall_miss))

    number_of_precision_miss = 0
    precision_misses = get_trigger_precision_misses(predictions, test_label, event_domain.get_event_type_index('None'), event_domain, examples)
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


def generate_argument_data_feature(generator, docs, extractor_name, model_flags, predicted_triggers=None):
    """
    :type generator: nlplingo.event.event_trigger.EventArgumentGenerator
    :type extractor_name: str
    :type docs: list[nlplingo.text.text_theory.Document]
    :type model_flags: dict
    """

    if model_flags.get('use_trigger', False):
        examples = generator.generate(docs, triggers=predicted_triggers)
    else:
        examples = generator.generate(docs)

    data = generator.examples_to_data_dict(examples)

    if extractor_name == 'event-argument_lstmcnn':
        if model_flags.get('use_event_embedding', False):
            data_list = [
                np.asarray(data['word_vec']),
                np.asarray(data['word_vec_forward']),
                np.asarray(data['word_vec_backward']),
                np.asarray(data['trigger_pos_array']),
                np.asarray(data['argument_pos_array']),
                np.asarray(data['lex']),
                np.asarray(data['event_array'])
            ]
        else:
            data_list = [
                np.asarray(data['word_vec']),
                np.asarray(data['word_vec_forward']),
                np.asarray(data['word_vec_backward']),
                np.asarray(data['trigger_pos_array']),
                np.asarray(data['argument_pos_array']),
                np.asarray(data['lex'])
            ]

    elif extractor_name == 'event-argument_lstm':
        print('Running_do_lstm')
        data_list = [
            np.asarray(data['word_vec_forward']),
            np.asarray(data['word_vec_backward'])
        ]

    elif extractor_name == 'event-argument_embedded':

        data_list = [
            np.asarray(data['lex']),
            #np.asarray(data['sent_vec']),
            np.asarray(data['event_array'])[:, 0],
        ]

        if model_flags.get('use_position_feat', False):
            data_list.extend(
                [
                    np.asarray(data['role_ner']),
                    np.asarray(data['rel_pos']),
                    np.asarray(data['lex_dist']),
                    np.asarray(data['is_unique_role']),
                    np.asarray(data['is_nearest_type']),
                ]
            )

        if model_flags.get('use_common_entity_name', False):
            data_list.append(np.asarray(data['common_word']))

        if model_flags.get('use_dep_emb', False):
            data_list.append(np.asarray(data['arg_trigger_dep']))
            print(data['arg_trigger_dep'])

    elif extractor_name == 'event-argument_cnn':
        if model_flags.get('do_dmcnn', False):
            data_list = [
                np.asarray(data['word_vec']),
                np.asarray(data['word_vec_left']),
                np.asarray(data['word_vec_middle']),
                np.asarray(data['word_vec_right']),
                np.asarray(data['trigger_pos_array']),
                np.asarray(data['trigger_pos_array_left']),
                np.asarray(data['trigger_pos_array_middle']),
                np.asarray(data['trigger_pos_array_right']),
                np.asarray(data['argument_pos_array']),
                np.asarray(data['argument_pos_array_left']),
                np.asarray(data['argument_pos_array_middle']),
                np.asarray(data['argument_pos_array_right']),
                np.asarray(data['event_array']),
                np.asarray(data['event_array_left']),
                np.asarray(data['event_array_middle']),
                np.asarray(data['event_array_right']),
                np.asarray(data['lex'])
            ]
        else:
            data_list = [np.asarray(data['word_vec'])]

            data_list.append(np.asarray(data['trigger_pos_array']))

            data_list.append(
                np.asarray(data['argument_pos_array'])
            )

            data_list.append(
                np.asarray(data['lex'])
            )

            if model_flags.get('use_event_embedding', False):
                data_list.append(np.asarray(data['event_array']))

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
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type argument_extractor: nlplingo.model.Extractor
    """
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)
    test_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings)

    for doc in train_docs:
        doc.apply_domain(argument_extractor.domain)
    for doc in test_docs:
        doc.apply_domain(argument_extractor.domain)

    generator = argument_extractor.generator

    argument_model = argument_extractor.extraction_model

    print('type(generator)={}'.format(type(generator)))
    print('type(argument_model)={}'.format(type(argument_model)))

    (train_examples, train_data, train_data_list, train_label) = generate_argument_data_feature(
        generator,
        train_docs,
        argument_extractor.model_type,
        argument_extractor.model_flags
    )

    print(train_label)

    # train_examples = generator.generate(train_docs)
    # train_data = generator.examples_to_data_dict(train_examples)
    # train_data_list = [np.asarray(train_data['word_vec']), np.asarray(train_data['pos_array']), np.asarray(train_data['event_array'])]
    # train_label = np.asarray(train_data['label'])
    #
    # print('#train_examples=%d' % (len(train_examples)))
    # print('train_data word_vec.len=%d pos_array.len=%d event_array.len=%d label.len=%d' % (
    #     len(train_data['word_vec']), len(train_data['pos_array']), len(train_data['event_array']),
    #     len(train_data['label'])))

    (test_examples, test_data, test_data_list, test_label) = generate_argument_data_feature(
        generator,
        test_docs,
        argument_extractor.model_type,
        argument_extractor.model_flags
    )

    # test_examples: list[event.event_argument.EventArgumentExample]

    # test_examples = generator.generate(test_docs)
    # test_data = generator.examples_to_data_dict(test_examples)
    # test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array']), np.asarray(test_data['event_array'])]
    # test_label = np.asarray(test_data['label'])
    #
    # print('#test_examples=%d' % (len(test_examples)))
    # print('test_data word_vec.len=%d pos_array.len=%d event_array.len=%d label.len=%d' % (
    #     len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['event_array']),
    #     len(test_data['label'])))

    # using predicted triggers to generate arg examples
    # test_examples_pt = generator.generate(test_docs, predicted_positive_triggers)
    # test_data_pt = generator.examples_to_data_dict(test_examples_pt)
    # test_data_list_pt = [np.asarray(test_data_pt['word_vec']), np.asarray(test_data_pt['pos_array']),
    #                   np.asarray(test_data_pt['event_array'])]
    # test_label_pt = np.asarray(test_data_pt['label'])

    #argument_model = MaxPoolEmbeddedRoleModel(params, event_domain, word_embeddings)
    argument_model.fit(train_data_list, train_label, test_data_list, test_label)

    # TODO this is probably stricter than the ACE way of scoring, which just requires that there's an entity-mention of same offsets with same role

    predictions = argument_model.predict(test_data_list)

    # calculate the recall denominator
    # number_test_args = 0
    # for doc in test_docs:
    #     for event in doc.events:
    #         number_test_args += event.number_of_arguments()

    #score = evaluate_f1(predictions, test_label, event_domain.get_event_role_index('None'), num_true=number_test_args)
    #print('Arg-score1: ' + score.to_string())

    score, score_breakdown, gold_labels = evaluate_arg_f1(argument_extractor.domain, test_label, test_examples, predictions)
    print('Arg-score: ' + score.to_string())

    for index, f1_score in score_breakdown.items():
        er = argument_extractor.domain.get_event_role_from_index(index)
        print('{}\t{}'.format(er, f1_score.to_string()))

    # predictions_pt = argument_model.predict(test_data_list_pt)
    # print('==== Calculate F1 for arg (using predicted triggers) ====')
    # # TODO we need to get the recall denominator correct
    # evaluate_f1(predictions_pt, test_label_pt, event_domain.get_event_role_index('None'))

    with open(params['train.score_file'], 'w') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            er = argument_extractor.domain.get_event_role_from_index(index)
            f.write('{}\t{}\n'.format(er, f1_score.to_string()))

    print('==== Saving Argument model ====')
    if params['save_model']:
        argument_model.save_keras_model(extractor.model_file)
    #with open(os.path.join(output_dir, 'argument.pickle'), u'wb') as f:
    #    pickle.dump(argument_model, f)

    # print('==== Loading model ====')
    # with open('role.try.pickle', u'rb') as f:
    #     model = pickle.load(f)
    #     """:type: model.cnn.EventExtractionModel"""
    #     model.load_keras_model(filename='role.try.hdf')
    #     predictions = model.predict(test_data_list)
    #     print('==== Model loading, calculating F1 ====')
    #     evaluate_f1(predictions, test_label, event_domain.get_event_role_index('None'))


def test_argument(params, word_embeddings, trigger_extractor, argument_extractor, scoring_domain=None):
    """
    :type params: dict
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type scoring_domain: nlplingo.event.event_domain.EventDomain
    """
    event_domain = argument_extractor.domain
    arg_generator = argument_extractor.generator
    trigger_generator = trigger_extractor.generator

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)
    for doc in test_docs:
        doc.apply_domain(event_domain)

    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_generator,
        test_docs,
        trigger_extractor.model_type,
        trigger_extractor.model_flags
    )
    (arg_examples, arg_data, arg_data_list, arg_label) = generate_argument_data_feature(
        arg_generator,
        test_docs,
        argument_extractor.model_type,
        argument_extractor.model_flags
    )
    for v in arg_data_list:
        print(v)


    # test_examples = generator.generate(test_docs)
    # test_data = generator.examples_to_data_dict(test_examples)
    # test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array']),
    #                   np.asarray(test_data['event_array'])]
    # test_label = np.asarray(test_data['label'])
    #
    # print('#test_examples=%d' % (len(test_examples)))
    # print('test_data word_vec.len=%d pos_array.len=%d event_array.len=%d label.len=%d' % (
    #     len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['event_array']),
    #     len(test_data['label'])))



    print('==== Loading Trigger model ====')
    trigger_predictions = trigger_extractor.extraction_model.predict(trigger_data_list)
    predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples,
                                                                  event_domain.get_event_type_index('None'),
                                                                  event_domain)

    print('==== Loading Argument model ====')
    argument_predictions = argument_extractor.extraction_model.predict(arg_data_list)

    score_with_gold_triggers, score_breakdown_with_gold_triggers, gold_labels = evaluate_arg_f1(
        event_domain, arg_label, arg_examples, argument_predictions, scoring_domain)
    #score_with_gold_triggers = evaluate_f1(argument_predictions, arg_label, event_domain.get_event_role_index('None'))
    print('Arg-scores with gold-triggers: {}'.format(score_with_gold_triggers.to_string()))

    for index, f1_score in score_breakdown_with_gold_triggers.items():
        er = event_domain.get_event_role_from_index(index)
        print('Arg-scores with gold-triggers: {}\t{}'.format(er, f1_score.to_string()))

    # generate arguments with predicted triggers
    (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
        generate_argument_data_feature(
            arg_generator,
            test_docs,
            argument_extractor.model_type,
            argument_extractor.model_flags,
            predicted_triggers=predicted_positive_triggers
        )

    # decode arguments with predicted triggers
    argument_predictions_pt = argument_extractor.extraction_model.predict(arg_data_list_pt)

    # evaluate arguments with predicted triggers
    score_with_predicted_triggers, score_breakdown_with_predicted_triggers, pred_labels = \
        evaluate_arg_f1(
            event_domain,
            arg_label_pt,
            arg_examples_pt,
            argument_predictions_pt,
            scoring_domain,
            gold_labels=gold_labels
        )
    #score_with_predicted_triggers.num_true = score_with_gold_triggers.num_true
    #score_with_predicted_triggers.calculate_score()
    #for role in score_breakdown_with_predicted_triggers:
    #    score_breakdown_with_predicted_triggers[role].num_true =
    # score_breakdown_with_gold_triggers[role].num_true
    #    score_breakdown_with_predicted_triggers[role].calculate_score()
    #score_with_predicted_triggers = evaluate_f1(argument_predictions_pt, arg_label_pt,
    #                                            event_domain.get_event_role_index('None'),
    #                                            num_true=score_with_gold_triggers.num_true)
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


def decode_trigger_argument(params, word_embeddings, extractors, argument_extractors):
    """
    :type params: dict
    :type word_embeddings: nlplingo.embeddings.WordEmbedding
    :type extractors: list[nlplingo.model.extractor.Extractor] # trigger extractors
    :type argument_extractors: list[nlplingo.model.extractor.Extractor] # argument extractors
    """
    # Find the trigger extractor

    trigger_extractor = None
    if len(extractors) > 1:
        raise RuntimeError('More than one trigger model cannot be used in decoding.')
    elif len(extractors) == 1:
        trigger_extractor = extractors[0]

    if len(argument_extractors) == 0:
        raise RuntimeError('At least one argument extractor must be specified to decode over arguments. {}'.format(len(extractors)))

    if trigger_extractor is None:
        raise RuntimeError('Trigger extractor must be specified in parameter file.')

    trigger_generator = trigger_extractor.generator

    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings)

    (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
        trigger_generator,
        test_docs,
        trigger_extractor.model_type,
        trigger_extractor.model_flags
    )

    predictions_output_file = params['predictions_file']
    clusters = {}

    print('==== Loading Trigger model ====')
    trigger_model = load_trigger_modelfile(trigger_extractor.model_file)
    predicted_positive_triggers = []
    if len(trigger_examples) > 0:
        trigger_predictions = trigger_model.predict(trigger_data_list)
        predicted_positive_triggers = get_predicted_positive_triggers(
            trigger_predictions,
            trigger_examples,
            trigger_extractor.domain.get_event_type_index('None'),
            trigger_extractor.domain)

    for docid in predicted_positive_triggers:
        for t in predicted_positive_triggers[docid]:
            """:type: nlplingo.event.event_trigger.EventTriggerExample"""
            print('PREDICTED-ANCHOR {} {} {} {} {}'.format(t.sentence.docid, t.event_type, '%.4f' % t.score, t.anchor.start_char_offset(), t.anchor.end_char_offset()))
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

    for extractor in argument_extractors:
        print('Loading argument model {}'.format(extractor.model_file))
        argument_model = load_argument_modelfile(extractor.model_file)

        if len(predicted_positive_triggers) > 0:
            # generate arguments with predicted triggers
            (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                generate_argument_data_feature(
                    extractor.generator,
                    test_docs,
                    extractor.model_type,
                    extractor.model_flags,
                    predicted_triggers=predicted_positive_triggers
                )

            pred_arg_max = []
            if len(arg_examples_pt) > 0:
                # decode arguments with predicted triggers
                argument_predictions_pt = argument_model.predict(arg_data_list_pt)
                pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

            for i, predicted_label in enumerate(pred_arg_max):
                #if predicted_label != event_domain.get_event_role_index('None'):
                if predicted_label != extractor.domain.get_event_role_index('None'):
                    eg = arg_examples_pt[i]
                    """:type: nlplingo.event.event_argument.EventArgumentExample"""
                    eg.score = argument_predictions_pt[i][predicted_label]
                    #predicted_role = event_domain.get_event_role_from_index(predicted_label)
                    predicted_role = extractor.domain.get_event_role_from_index(predicted_label)

                    if predicted_role == 'Time' and eg.argument.label != 'TIMEX2.TIME':
                        continue
                    if predicted_role == 'Place' and not (eg.argument.label.startswith('GPE') or eg.argument.label.startswith(
                            'FAC') or eg.argument.label.startswith('LOC') or eg.argument.label.startswith('ORG')):
                        continue
                    if predicted_role == 'Actor' and not (eg.argument.label.startswith('PER') or eg.argument.label.startswith(
                            'ORG') or eg.argument.label.startswith('GPE')):
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
                    # if eg.sentence.docid == 'ENG_NW_NODATE_0001':
                    #     print("predicted_role:{},current_anchor:{},current_word:{},array:{}".format(predicted_role,eg.anchor.text,eg.argument.text,[tokenIdx.index_in_sentence for tokenIdx in eg.argument.tokens]))
                    trigger[predicted_role] = argument

    with open(predictions_output_file, 'w') as fp:
            json.dump(clusters, fp, indent=4, sort_keys=True)


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



# =============== Event sentence =================
def get_predicted_positive_sentences(predictions, examples, none_class_index, event_domain):
    """Collect the predicted positive sentences and organize them by docid
    Also, use the predicted event_type for each such sentence example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.event.event_sentence.EventSentenceExample]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    assert len(predictions)==len(examples)
    ret = defaultdict(list)

    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            eg = examples[i]
            """:type: nlplingo.event.event_sentence.EventSentenceExample"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            ret[eg.sentence.docid].append(eg)
    return ret

def generate_sentence_data_feature(generator, docs):
    """
    :type generator: nlplingo.event.event_sentence.EventSentenceGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    """
    examples = generator.generate(docs)
    data = generator.examples_to_data_dict(examples)
    data_list = [np.asarray(data['word_vec'])]
    label = np.asarray(data['label'])

    print('#sentence-examples=%d' % (len(examples)))
    print('data word_vec.len=%d label.len=%d' % (len(data['word_vec']), len(data['label'])))
    return (examples, data, data_list, label)

def sentence_modeling(params, train_docs, test_docs, event_domain, word_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    """
    generator = EventSentenceGenerator(event_domain, params)

    (train_examples, train_data, train_data_list, train_label) = generate_sentence_data_feature(generator, train_docs)

    # train_examples = generator.generate(train_docs)
    # train_data = generator.examples_to_data_dict(train_examples)
    # train_data_list = [np.asarray(train_data['word_vec']), np.asarray(train_data['pos_array'])]
    # train_label = np.asarray(train_data['label'])
    #
    # print('#train_examples=%d' % (len(train_examples)))
    # print('train_data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    #     len(train_data['word_vec']), len(train_data['pos_array']), len(train_data['label'])))

    (test_examples, test_data, test_data_list, test_label) = generate_sentence_data_feature(generator, test_docs)

    # test_examples = generator.generate(test_docs)
    # test_data = generator.examples_to_data_dict(test_examples)
    # test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array'])]
    # test_label = np.asarray(test_data['label'])
    #
    # print('#test_examples=%d' % (len(test_examples)))
    # print('test_data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    #     len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['label'])))

    sentence_model = MaxPoolEmbeddedSentenceModel(params, event_domain, word_embeddings)
    sentence_model.fit(train_data_list, train_label, test_data_list, test_label)

    predictions = sentence_model.predict(test_data_list)
    predicted_positive_sentences = get_predicted_positive_sentences(predictions, test_examples, event_domain.get_event_type_index('None'), event_domain)

    # calculate the recall denominator
    number_test_positives = 0
    for doc in test_docs:
        for sentence in doc.sentences:
            labels = set()
            for event in sentence.events:
                labels.add(event.label)
            number_test_positives += len(labels)

    score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'), num_true=number_test_positives)
    print(score.to_string())

    output_dir = params.get_string('output_dir')
    with open(os.path.join(output_dir, 'train_sentence.score'), 'w') as f:
        f.write(score.to_string() + '\n')

    print('==== Saving Sentence model ====')
    sentence_model.save_keras_model(os.path.join(output_dir, 'sentence.hdf'))
    with open(os.path.join(output_dir, 'sentence.pickle'), u'wb') as f:
        pickle.dump(sentence_model, f)

    return predicted_positive_sentences


def train_sentence(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    train_docs = prepare_docs(params.get_string('filelist.train'))
    test_docs = prepare_docs(params.get_string('filelist.dev'))

    predicted_positive_sentences = sentence_modeling(params, train_docs, test_docs, event_domain, word_embeddings)


def to_example_pairs(examples, data, data_list, label):
    """
    :type examples: list[nlplingo.event.event_sentence.EventSentenceExample]
    :type data: defaultdict(list)
    :type data_list: [np.asarray(list)]
    :type label: np.asarray(list)
    """
    eg_by_label = defaultdict(list)
    for eg in examples:
        eg_by_label[eg.event_type].append(eg)

    for label in eg_by_label.keys():
        n = len(eg_by_label[label])
        n_choose_2 = (n * (n-1))/2
        print('{} combinations for label {}'.format(n_choose_2, label))

def generate_pair_data_from_triggers_pairs(egs1, egs2, pair_generator):
    """
    :type egs1: list[nlplingo.event.event_trigger.EventTriggerExample]
    :type egs2: list[nlplingo.event.event_trigger.EventTriggerExample]
    :type pair_generator: nlplingo.event.event_pair.EventPairGenerator
    """

    pair_examples = pair_generator.generate_cross_product(egs1, egs2)

    data = pair_generator.examples_to_data_dict(pair_examples)
    data_list = [np.asarray(data['word_vec1']), np.asarray(data['word_vec2']),
                 np.asarray(data['pos_data1']), np.asarray(data['pos_data2'])]
                 #np.asarray(data['word_cvec1']), np.asarray(data['word_cvec2']),
                 #np.asarray(data['dep_vec1']), np.asarray(data['dep_vec2'])]
    label = np.asarray(data['label'])
    #label = k_utils.to_categorical(np.array(data['label']), num_classes=2)

    print('data word_vec1.len=%d word_vec2.len=%d label.len=%d' % (len(data['word_vec1']), len(data['word_vec2']), len(data['label'])))
    return EventPairData(None, pair_examples, data, data_list, label)
    #return (pair_examples, data, data_list, label)

def generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, docs, training=False):
    """
    We set include_none=False during training, include_none=True when it is test data

    :type novel_event_type: nlplingo.event.novel_event_type.NovelEventType
    :type trigger_generator: nlplingo.event.event_trigger.EventTriggerGenerator
    :type pair_generator: nlplingo.event.event_pair.EventPairGenerator
    :type docs: list[nlplingo.text.text_theory.Document]
    """

    trigger_examples = trigger_generator.generate(docs)
    """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""

    egs = None
    """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""
    if training:
        egs = novel_event_type.filter_train(trigger_examples)
        et_count = defaultdict(int)
        for eg in egs:
            et_count[eg.event_type] += 1
        for et in et_count:
            print('In train egs: {} {}'.format(et, et_count[et]))

        pair_examples = pair_generator.generate_train(egs)
    else:
        #egs = novel_event_type.filter_test(trigger_examples)
        egs = trigger_examples
        print('novel_event_type.filter_test #trigger_examples={} #egs={}'.format(len(trigger_examples), len(egs)))
        et_count = defaultdict(int)
        for eg in egs:
            et_count[eg.event_type] += 1
        for et in et_count:
            print('In test egs: {} {}'.format(et, et_count[et]))

        pair_examples = pair_generator.generate_test(egs)


    data = pair_generator.examples_to_data_dict(pair_examples)
    data_list = [np.asarray(data['word_vec1']), np.asarray(data['word_vec2']),
                 np.asarray(data['pos_data1']), np.asarray(data['pos_data2'])]
                 #np.asarray(data['word_cvec1']), np.asarray(data['word_cvec2']),
                 #np.asarray(data['dep_vec1']), np.asarray(data['dep_vec2'])]
    label = np.asarray(data['label'])
    #label = k_utils.to_categorical(np.array(data['label']), num_classes=2)

    print('data word_vec1.len=%d word_vec2.len=%d label.len=%d' % (len(data['word_vec1']), len(data['word_vec2']), len(data['label'])))

    return EventPairData(egs, pair_examples, data, data_list, label)
    #return (egs, pair_examples, data, data_list, label)


def calculate_pair_true_positive(docs, dataset_prefix, target_event_types):
    """We need this because in generating trigger candidates, we heuristically reject some candidates
    So we cannot calculate the true positive from the candidates. We need to go back to the doc level annotations.

    :type docs: list[nlplingo.text.text_theory.Document]
    :type dataset_prefix: str
    :type target_event_types: set[str]
    """
    event_count = defaultdict(int)
    for doc in docs:
        for event in doc.events:
            if event.label in target_event_types:
                event_count[event.label] += 1
    pair_true_positive = 0
    for et in event_count:
        print('In {} docs: {} {}'.format(dataset_prefix, et, event_count[et]))
        count = event_count[et]
        pair_true_positive += (count * (count - 1)) / 2
    return pair_true_positive

def pair_modeling(params, train_docs, dev_docs, test_docs, event_domain, word_embeddings, causal_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type dev_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type causal_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    """

    novel_event_type = NovelEventType(params)

    trigger_generator = EventTriggerGenerator(event_domain, params)
    pair_generator = EventPairGenerator(event_domain, params, word_embeddings)


    print('#### Generating Training data')
    #(train_triggers, train_examples, train_data, train_data_list, train_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, train_docs, training=True)
    train_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, train_docs, training=True)
    # train_triggers: list[nlplingo.event.event_trigger.EventTriggerExample]

    train_triggers_new_types = [eg for eg in train_data.trigger_examples if eg.event_type in novel_event_type.new_types]
    """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""

    print('#### Generating Dev data')
    #(dev_triggers, dev_examples, dev_data, dev_data_list, dev_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, dev_docs, training=False)
    dev_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, dev_docs, training=False)

    print('#### Generating Test data')
    #(test_triggers, test_examples, test_data, test_data_list, test_label) = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, test_docs, training=False)
    test_data = generate_pair_data_feature(novel_event_type, trigger_generator, pair_generator, test_docs, training=False)

    # The idea is in training data, we have annotated very few trigger examples of the new event types
    # We want to take the cross product between these few training examples, and the test trigger candidates
    # Later on, we can then feed the pairwise probabilities into some heuristics to select test trigger candidates that are most similar to the traininig examples
    train_test_data = generate_pair_data_from_triggers_pairs(train_triggers_new_types, test_data.trigger_examples, pair_generator)

    pair_model = MaxPoolEmbeddedPairModel(params, event_domain, word_embeddings, causal_embeddings)
    print('** train_data_list')
    print(train_data.data_list)
    print('** train_label')
    print(train_data.label)
    pair_model.fit(train_data.data_list, train_data.label, dev_data.data_list, dev_data.label)



    #print(predictions)

    #print('len(test_label)={}'.format(len(test_label)))
    #print('len(predictions)={}'.format(len(predictions)))

    #accuracy = evaluate_accuracy(predictions, test_label)
    #print('Accuracy=%.2f' % (accuracy))

    # class_labels = []
    # for eg in test_examples:
    #     if eg.label_string == 'SAME':
    #         class_labels.append(eg.eg1.event_type)
    #     else:
    #         class_labels.append('None')





    #### score on dev
    dev_predictions = pair_model.predict(dev_data.data_list)
    dev_tp_existing_type = calculate_pair_true_positive(dev_docs, 'dev', novel_event_type.existing_types)
    dev_tp_new_type = calculate_pair_true_positive(dev_docs, 'dev', novel_event_type.new_types)

    f1_dict_existing = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, novel_event_type.existing_types, dev_tp_existing_type)
    for f1 in f1_dict_existing:
        print('Dev Existing-type F1 score: {}'.format(f1.to_string()))
    f1_dict_new = evaluate_f1_binary(dev_predictions, dev_data.label, dev_data.pair_examples, novel_event_type.new_types, dev_tp_new_type)
    for f1 in f1_dict_new:
        print('Dev New-type F1 score: {}'.format(f1.to_string()))
    with open(params.get_string('output_dir') + '/dev.score', 'w') as f:
        for f1 in f1_dict_existing:
            f.write('Existing-type F1 score: {}\n'.format(f1.to_string()))
        for f1 in f1_dict_new:
            f.write('New-type F1 score: {}\n'.format(f1.to_string()))

    #### score on test
    test_predictions = pair_model.predict(test_data.data_list)
    test_tp_existing_type = calculate_pair_true_positive(test_docs, 'test', novel_event_type.existing_types)
    test_tp_new_type = calculate_pair_true_positive(test_docs, 'test', novel_event_type.new_types)

    f1_dict_existing = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, novel_event_type.existing_types, test_tp_existing_type)
    for f1 in f1_dict_existing:
        print('Test Existing-type F1 score: {}'.format(f1.to_string()))
    f1_dict_new = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, novel_event_type.new_types, test_tp_new_type)
    for f1 in f1_dict_new:
        print('Test New-type F1 score: {}'.format(f1.to_string()))
    f1_test_lines = []
    for et in novel_event_type.new_types:
        tp = calculate_pair_true_positive(test_docs, 'test', et)
        f1 = evaluate_f1_binary(test_predictions, test_data.label, test_data.pair_examples, set([et]), tp, et)
        f1_test_lines.append(f1[0].to_string())
    with open(params.get_string('output_dir') + '/test.score', 'w') as f:
        for f1 in f1_dict_existing:
            f.write('Existing-type F1 score: {}\n'.format(f1.to_string()))
        for f1 in f1_dict_new:
            f.write('New-type F1 score: {}\n'.format(f1.to_string()))
        for score in f1_test_lines:
            f.write('F1 score: {}\n'.format(score))

    #### score on train-test
    train_test_predictions = pair_model.predict(train_test_data.data_list)
    f1_train_test_lines = []
    f1_train_test = evaluate_f1_binary(train_test_predictions, train_test_data.label, train_test_data.pair_examples, novel_event_type.new_types, None)
    f1_train_test_lines.append(f1_train_test[0].to_string())
    for et in novel_event_type.new_types:
        f1 = evaluate_f1_binary(train_test_predictions, train_test_data.label, train_test_data.pair_examples, set([et]), None, et)
        f1_train_test_lines.append(f1[0].to_string())
    with open(params.get_string('output_dir') + '/train_test.score', 'w') as f:
        for score in f1_train_test_lines:
            f.write('F1 score: {}\n'.format(score))
    print_pair_predictions(train_test_data.pair_examples, train_test_predictions, params.get_string('output_dir')+'/train_test.predictions')

                # output_dir = params.get_string('output_dir')
    # with open(os.path.join(output_dir, 'train_pair.score'), 'w') as f:
    #     f.write('Accuracy={}\n'.format(accuracy))
    #
    # print('==== Saving Pair model ====')
    # pair_model.save_keras_model(os.path.join(output_dir, 'pair.hdf'))
    # with open(os.path.join(output_dir, 'pair.pickle'), u'wb') as f:
    #     pickle.dump(pair_model, f)


def train_pair(params, word_embeddings, event_domain):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    train_docs = prepare_docs(params.get_string('filelist.train'), word_embeddings)
    dev_docs = prepare_docs(params.get_string('filelist.dev'), word_embeddings)
    test_docs = prepare_docs(params.get_string('filelist.test'), word_embeddings)

    pair_modeling(params, train_docs, dev_docs, test_docs, event_domain, word_embeddings)


def generate_event_statistics(params, word_embeddings):
    """
    :type params: nlplingo.common.parameters.Parameters
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type event_domain: nlplingo.event.event_domain.EventDomain
    """
    docs = prepare_docs(params['data']['train']['filelist'], word_embeddings)

    anchor_types = defaultdict(int)
    anchor_lines = []
    for doc in docs:
        for sent in doc.sentences:
            for event in sent.events:
                anchor = event.anchors[0]
                anchor_types[anchor.label] += 1
                anchor_lines.append('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(doc.docid, anchor.id, anchor.label, anchor.text, anchor.head().text, anchor.start_char_offset(), anchor.end_char_offset()))

    with codecs.open(params['output.event_type_count'], 'w', encoding='utf-8') as o:
        for et in sorted(anchor_types):
            o.write('{}\t{}\n'.format(et, anchor_types[et]))

    with codecs.open(params['output.anchor_info'], 'w', encoding='utf-8') as o:
        for l in anchor_lines:
            o.write(l)
            o.write('\n')


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)   # train_trigger, train_arg, test_trigger, test_arg
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    #params = Parameters(args.params)
    #params.print_params()


    embeddings = dict()

    # load word embeddings
    embeddings_params = params['embeddings']
    word_embeddings = WordEmbeddingFactory.createWordEmbedding(
        embeddings_params.get('type', 'word_embeddings'),
        embeddings_params
    )

    embeddings['word_embeddings'] = word_embeddings

    if 'dependency_embeddings' in params:
        dep_embeddings_params = params['dependency_embeddings']
        dependency_embeddings = WordEmbeddingFactory.createWordEmbedding(
            dep_embeddings_params.get('type', 'dependency_embeddings'),
            dep_embeddings_params
        )
        embeddings['dependency_embeddings'] = dependency_embeddings

    print('Embeddings loaded')
    #if params.has_key('causal_embedding.embedding_file'):
    #    causal_embeddings = WordEmbedding(params, params.get_string('causal_embedding.embedding_file'),
    #                                params.get_int('causal_embedding.vocab_size'), params.get_int('causal_embedding.vector_size'))
    #else:
    #    causal_embeddings = None

    load_extractor_models_from_file = False
    if args.mode in {
        'test_trigger',
        'test_argument',
        'decode_trigger_argument',
        'decode_trigger'
    }:
        load_extractor_models_from_file = True

    argument_extractors = []

    # TODO: change this to trigger_extractors
    # TODO: support multiple trigger extractors
    extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractor = Extractor(
            params,
            extractor_params,
            embeddings,
            load_extractor_models_from_file
        )
        if extractor.model_type.startswith('event-trigger_'):
            extractors.append(extractor)
        elif extractor.model_type.startswith('event-argument_'):
            argument_extractors.append(extractor)
        else:
            raise RuntimeError(
                'Extractor model type: {} not implemented.'.format(
                    extractor.model_type
                )
            )

    # event_domain = None
    # if params.get_string('domain') == 'general':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), params.get_string('domain'))
    # elif params.get_string('domain') == 'cyber':
    #     event_domain = CyberDomain()
    # elif params.get_string('domain') == 'ace':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ace')
    # elif params.get_string('domain') == 'precursor':
    #     event_domain = PrecursorDomain()
    # elif params.get_string('domain') == 'ui':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')
    # elif params.get_string('domain') == 'ace-precursor':
    #     event_domain = AcePrecursorDomain()
    # elif params.get_string('domain') == 'cyberattack':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')

    if 'domain_ontology.scoring' in params:
        scoring_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology.scoring'), 'scoring')
    else:
        scoring_domain = None


    #print(event_domain.to_string())

    if args.mode == 'train_trigger_from_file':
        train_trigger_from_file(params, embeddings, extractors[0])
    elif args.mode == 'train_trigger_from_model':
        train_trigger_from_file(params, word_embeddings, extractors[0], from_model=True)

    elif args.mode == 'train_trigger_from_feature':
        train_trigger_from_feature(params, extractors[0])
    elif args.mode == 'test_trigger':
        test_trigger(params, embeddings, extractors[0])

    elif args.mode == 'train_argument':
        train_argument(params, embeddings, argument_extractors[0])
    elif args.mode == 'test_trigger_list':
        test_trigger_list(params, word_embeddings, event_domain)
    elif args.mode == 'test_argument':
        test_argument(params, embeddings, extractors[0], argument_extractors[0], scoring_domain)
    elif args.mode == 'train_sentence':
        train_sentence(params, word_embeddings, event_domain)
    elif args.mode == 'train_pair':
        train_pair(params, word_embeddings, event_domain)
    elif args.mode == 'decode_trigger_argument':
        decode_trigger_argument(params, embeddings, extractors, argument_extractors)
    elif args.mode == 'decode_trigger':
        decode_trigger(params, word_embeddings, extractors[0])
    elif args.mode == 'event_statistics':
        generate_event_statistics(params, word_embeddings)
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))