from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import codecs
import spacy

import jsonpickle
import pickle

from collections import defaultdict

import numpy as np

from nlplingo.text.text_theory import Document
from nlplingo.annotation.idt import process_idt_file
from nlplingo.annotation.enote import process_enote_file
from nlplingo.embeddings.word_embeddings import WordEmbedding
from nlplingo.event.event_trigger import EventTriggerGenerator
from nlplingo.event.event_argument import EventArgumentGenerator
from nlplingo.event.event_domain import CyberDomain
import nlplingo.common.utils as text_utils
from nlplingo.common.parameters import Parameters

from nlplingo.model.event_cnn import MaxPoolEmbeddedTriggerModel
from nlplingo.model.event_cnn import MaxPoolEmbeddedRoleModel
from nlplingo.model.event_cnn import evaluate_f1


def parse_filelist_line(line):
    text_file = None
    idt_file = None
    enote_file = None

    for file in line.strip().split():
        if file.startswith('TEXT:'):
            text_file = file[len('TEXT:'):]
        elif file.startswith('IDT:'):
            idt_file = file[len('IDT:'):]
        elif file.startswith('ENOTE:'):
            enote_file = file[len('ENOTE:'):]

    if text_file is None:
        raise ValueError('text file must be present!')
    return (text_file, idt_file, enote_file)

def read_docs(filelists):
    """
    :type filelists: list[str]
    Returns:
        list[nlplingo.text.text_theory.Document]
    """
    docs = []
    """:type docs: list[text.text_theory.Document]"""
    for line in filelists:
        (text_file, idt_file, enote_file) = parse_filelist_line(line)

        docid = os.path.basename(text_file)

        text_f = codecs.open(text_file, 'r', encoding='utf-8')
        all_text = text_f.read()
        text_f.close()
        doc = Document(docid, all_text.strip())

        if idt_file is not None:
            doc = process_idt_file(doc, idt_file)  # adds entity mentions

        if enote_file is not None:
            doc = process_enote_file(doc, enote_file, auto_adjust=True)  # adds events

        docs.append(doc)
    return docs

def docs_to_jsonfile(docs, filename):
    """
    :type docs: list[nlplingo.text.text_theory.Document]
    :type filename: str
    """
    with codecs.open(filename, 'w', encoding='utf-8') as outf:
        outf.write(jsonpickle.encode(docs))

def jsonfile_to_docs(filename):
    """
    :type filename: str
    """
    f = codecs.open(filename, 'r', encoding='utf-8')
    docs = jsonpickle.decode(f.read())
    """:type: list[text.text_theory.Document"""
    f.close()
    return docs

def get_predicted_positive_triggers(predictions, examples, none_class_index, event_domain):
    """Collect the predicted positive triggers and organize them by docid
    Also, use the predicted event_type for each such trigger example

    :type predictions: numpy.nparray
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    :type event_domain: event.event_domain.EventDomain
    """
    assert len(predictions)==len(examples)
    ret = defaultdict(list)

    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            eg = examples[i]
            """:type: event.event_trigger.EventTriggerExample"""
            eg.event_type = event_domain.get_event_type_from_index(index)
            ret[eg.sentence.docid].append(eg)
    return ret

def trigger_modeling(params, train_docs, test_docs, event_domain, word_embeddings):
    """
    :type params: common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    """
    generator = EventTriggerGenerator(event_domain, params)

    train_examples = generator.generate(train_docs)
    train_data = generator.examples_to_data_dict(train_examples)
    train_data_list = [np.asarray(train_data['word_vec']), np.asarray(train_data['pos_array'])]
    train_label = np.asarray(train_data['label'])

    print('#train_examples=%d' % (len(train_examples)))
    print('train_data word_vec.len=%d pos_array.len=%d label.len=%d' % (len(train_data['word_vec']), len(train_data['pos_array']), len(train_data['label'])))

    test_examples = generator.generate(test_docs)
    test_data = generator.examples_to_data_dict(test_examples)
    test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array'])]
    test_label = np.asarray(test_data['label'])

    print('#test_examples=%d' % (len(test_examples)))
    print('test_data word_vec.len=%d pos_array.len=%d label.len=%d' % (
    len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['label'])))

    trigger_model = MaxPoolEmbeddedTriggerModel(params, event_domain, word_embeddings)
    trigger_model.fit(train_data_list, train_label, test_data_list, test_label)

    predictions = trigger_model.predict(test_data_list)

    score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_type_index('None'))

    predicted_positive_triggers = get_predicted_positive_triggers(predictions, test_examples, event_domain.get_event_type_index('None'), event_domain)
    return predicted_positive_triggers

def argument_modeling(params, train_docs, test_docs, event_domain, word_embeddings, predicted_positive_triggers):
    """
    :type params: common.parameters.Parameters
    :type train_docs: list[nlplingo.text.text_theory.Document]
    :type test_docs: list[nlplingo.text.text_theory.Document]
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
    :type predicted_positive_triggers: list[nlplingo.event.event_trigger.EventTriggerExample]
    """
    generator = EventArgumentGenerator(event_domain, params)

    train_examples = generator.generate(train_docs)
    train_data = generator.examples_to_data_dict(train_examples)
    train_data_list = [np.asarray(train_data['word_vec']), np.asarray(train_data['pos_array']), np.asarray(train_data['event_array'])]
    train_label = np.asarray(train_data['label'])

    print('#train_examples=%d' % (len(train_examples)))
    print('train_data word_vec.len=%d pos_array.len=%d event_array.len=%d label.len=%d' % (
    len(train_data['word_vec']), len(train_data['pos_array']), len(train_data['event_array']), len(train_data['label'])))

    test_examples = generator.generate(test_docs)
    test_data = generator.examples_to_data_dict(test_examples)
    test_data_list = [np.asarray(test_data['word_vec']), np.asarray(test_data['pos_array']), np.asarray(test_data['event_array'])]
    test_label = np.asarray(test_data['label'])

    print('#test_examples=%d' % (len(test_examples)))
    print('test_data word_vec.len=%d pos_array.len=%d event_array.len=%d label.len=%d' % (
        len(test_data['word_vec']), len(test_data['pos_array']), len(test_data['event_array']), len(test_data['label'])))

    # using predicted triggers to generate arg examples
    test_examples_pt = generator.generate(test_docs, predicted_positive_triggers)
    test_data_pt = generator.examples_to_data_dict(test_examples_pt)
    test_data_list_pt = [np.asarray(test_data_pt['word_vec']), np.asarray(test_data_pt['pos_array']),
                      np.asarray(test_data_pt['event_array'])]
    test_label_pt = np.asarray(test_data_pt['label'])

    argument_model = MaxPoolEmbeddedRoleModel(params, event_domain, word_embeddings)
    argument_model.fit(train_data_list, train_label, test_data_list, test_label)

    # TODO this is probably stricter than the ACE way of scoring, which just requires that there's an entity-mention of same offsets with same role

    predictions = argument_model.predict(test_data_list)
    print('==== Calculate F1 for arg ====')
    score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_role_index('None'))

    predictions_pt = argument_model.predict(test_data_list_pt)
    print('==== Calculate F1 for arg (using predicted triggers) ====')
    # TODO we need to get the recall denominator correct
    score, score_breakdown = evaluate_f1(predictions_pt, test_label_pt, event_domain.get_event_role_index('None'))

    print('==== Saving model ====')
    argument_model.keras_model.save('role.try.hdf')
    with open('role.try.pickle', u'wb') as f:
        pickle.dump(argument_model, f)

    print('==== Loading model ====')
    with open('role.try.pickle', u'rb') as f:
        model = pickle.load(f)
        """:type: model.cnn.EventExtractionModel"""
        model.load_keras_model(filename='role.try.hdf')
        predictions = model.predict(test_data_list)
        print('==== Model loading, calculating F1 ====')
        score, score_breakdown = evaluate_f1(predictions, test_label, event_domain.get_event_role_index('None'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--params')
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    # read IDT and ENote annotations
    train_docs = read_docs(text_utils.read_file_to_list(args.train))
    test_docs = read_docs(text_utils.read_file_to_list(args.test))

    # load word embeddings
    word_embeddings = WordEmbedding(params)

    # apply Spacy for sentence segmentation and tokenization, using Spacy tokens to back Anchor and EntityMention
    spacy_en = spacy.load('en')
    for doc in train_docs:
        doc.annotate_sentences(word_embeddings, spacy_en)
    for doc in test_docs:
        doc.annotate_sentences(word_embeddings, spacy_en)

    number_anchors = 0
    number_args = 0
    number_assigned_anchors = 0
    number_assigned_args = 0
    for doc in train_docs:
        for event in doc.events:
            number_anchors += event.number_of_anchors()
            number_args += event.number_of_arguments()
        for sent in doc.sentences:
            for event in sent.events:
                number_assigned_anchors += event.number_of_anchors()
                number_assigned_args += event.number_of_arguments()
    print('In training documents, #anchors=%d #assigned_anchors=%d , #args=%d #assigned_args=%d' % \
          (number_anchors, number_assigned_anchors, number_args, number_assigned_args))

    number_anchors = 0
    number_args = 0
    number_assigned_anchors = 0
    number_assigned_args = 0
    for doc in test_docs:
        for event in doc.events:
            number_anchors += event.number_of_anchors()
            number_args += event.number_of_arguments()
        for sent in doc.sentences:
            for event in sent.events:
                number_assigned_anchors += event.number_of_anchors()
                number_assigned_args += event.number_of_arguments()
    print('In testing documents, #anchors=%d #assigned_anchors=%d , #args=%d #assigned_args=%d' % \
          (number_anchors, number_assigned_anchors, number_args, number_assigned_args))


    # initialize a particular event domain, which stores info on the event types and event roles
    event_domain = CyberDomain()

    predicted_positive_triggers = trigger_modeling(params, train_docs, test_docs, event_domain, word_embeddings)

    argument_modeling(params, train_docs, test_docs, event_domain, word_embeddings, predicted_positive_triggers)

    # trigger_generator = EventTriggerGenerator(event_domain)
    # train_trigger_examples = trigger_generator.generate(train_docs)
    # train_trigger_data = trigger_generator.examples_to_data_dict(train_trigger_examples)
    #
    #
    # argument_generator = EventArgumentGenerator(event_domain)
    # argument_generator.generate(docs)
    #
    # trigger_model = CNNTriggerModel(event_domain, word_embeddings)
    # trigger_data = trigger_generator.data_dict
    # train_data_list = [np.asarray(trigger_data['word_vec']), np.asarray(trigger_data['pos_array'])]
    # train_label = np.asarray(trigger_data['label'])
    # weight = np.ones(train_label.shape[0])
    # trigger_model.fit(train_data_list, train_label, train_data_list, train_label, sample_weight=weight)
