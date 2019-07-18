from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import pickle
from collections import defaultdict
import re
import codecs
import json
import argparse

import numpy as np
from future.builtins import range

from keras.models import load_model as keras_load_model

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import Struct
from nlplingo.common.utils import F1Score
from nlplingo.common.parameters import Parameters

from nlplingo.event.event_domain import EventDomain
from nlplingo.embeddings.word_embeddings import WordEmbedding

from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import Token
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_theory import Document

from nlplingo.model.wordpair_model import MaxPoolEmbeddedWordPairModel
from nlplingo.model.event_cnn import evaluate_f1


pair_labels = {'DIFFERENT':0, 'SAME':1}

def get_recall_misses(prediction, label, none_class_index, event_domain, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    """
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    assert len(label_arg_max) == len(pred_arg_max)
    assert len(label_arg_max) == len(examples)

    misses = defaultdict(int)
    for i in range(len(label_arg_max)):
        if label_arg_max[i] != none_class_index:
            et = event_domain.get_event_type_from_index(label_arg_max[i])
            if pred_arg_max[i] != label_arg_max[i]:  # recall miss
                eg = examples[i]
                anchor_string = '_'.join(token.text.lower() for token in eg.anchor.tokens)
                misses['{}\t({})'.format(et, anchor_string)] += 1
    return misses


def get_precision_misses(prediction, label, none_class_index, event_domain, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    """
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    assert len(label_arg_max) == len(pred_arg_max)
    assert len(label_arg_max) == len(examples)

    misses = defaultdict(int)
    for i in range(len(pred_arg_max)):
        if pred_arg_max[i] != none_class_index:
            et = event_domain.get_event_type_from_index(pred_arg_max[i])
            if pred_arg_max[i] != label_arg_max[i]:  # precision miss
                eg = examples[i]
                anchor_string = '_'.join(token.text.lower() for token in eg.anchor.tokens)
                misses['{}\t({})'.format(et, anchor_string)] += 1
    return misses


class WordPairGenerator(object):
    # we only accept tokens of the following part-of-speech categories as trigger candidates
    # trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
    trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])

    def __init__(self, params, word_embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        """
        self.params = params
        self.word_embeddings = word_embeddings
        self.max_sent_length = params.get_int('max_sent_length')
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        self.embedding_prefix = params.get_string('embedding.prefix')
        print(self.embedding_prefix)
        self.statistics = defaultdict(int)

    def generate_documents(self, json_file, word_embeddings):
        """
        :type json_file: str
        :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """

        with codecs.open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        # as we read in each sentence and its associated word-constraint examples,
        # collate them into documents
        docs_sentences_text = dict()  # [docid] -> dict(sent-id, sent-text)
        for data in datas:
            guid = data['guid']
            docid = WordPairGenerator._docid_from_guid(guid)

            sentence_text = data['sentence']
            if docid in docs_sentences_text:
                docs_sentences_text[docid][guid] = sentence_text
            else:
                d = dict()
                d[guid] = sentence_text
                docs_sentences_text[docid] = d

        docs = []
        """:type: list[nlplingo.text.text_theory.Document]"""
        for docid in docs_sentences_text:
            d = docs_sentences_text[docid]
            doc = self._doc_from_sentence_texts(docid, [(sent_id, sent_text) for sent_id, sent_text in d.items()])
            docs.append(doc)

        for doc in docs:
            for sent in doc.sentences:
                doc.annotate_sentence_with_word_embeddings(sent, word_embeddings, None, self.embedding_prefix)

        print('Read in {} documents'.format(len(docs)))
        return docs

    def generate_examples(self, json_file, docs):
        """
        :type json_file: str
        :type docs: dict[str, nlplingo.text.text_theory.Document]
        :rtype: list[nlplingo.event.wordpair.WordPairExample]
        """
        ret = []
        """:type: list[nlplingo.event.wordpair.WordPairExample]"""

        with codecs.open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        counter = 0
        for data in datas:
            guid = data['guid']
            docid = WordPairGenerator._docid_from_guid(guid)
            if docid not in docs.keys():    # this might be true, if this doc contains only overly sentences which were all skipped
                continue
            else:
                doc = docs[docid]

            sentence = None
            """:type: nlplingo.text.text_span.Sentence"""
            for sent in doc.sentences:
                if sent.sent_id == guid:
                    sentence = sent
                    break
            if sentence is None:    # we skip overly long sentences, so this might be None
                continue

            start_token_index = data['start_token_index']
            end_token_index = data['end_token_index']
            if start_token_index != end_token_index:
                continue

            assert start_token_index < sentence.number_of_tokens()
            token = sentence.tokens[start_token_index]

            for annotation in data['examples']:
                word = annotation['word']
                constraint = annotation['constraint']
                if annotation['label'] == 1:
                    label = 'SAME'
                else:
                    label = 'DIFFERENT'
                lexsn = annotation['lexsn']

                anchor_id = '{}[{}-{}]'.format(guid, start_token_index, end_token_index)
                anchor = Anchor(anchor_id, IntPair(token.start_char_offset(), token.end_char_offset()), token.text, label)
                anchor.with_tokens([token])

                example = WordPairExample(anchor, sentence, params, label, word, constraint)
                self._generate_example(example, [None] + sentence.tokens, self.max_sent_length, self.neighbor_dist, self.embedding_prefix)
                if self._use_example(example):
                    ret.append(example)
                else:
                    self.statistics['Skipping example, where word or constraint has no embeddings'] += 1

            counter += 1
            if (counter % 1000) == 0:
                print('Generated examples for {} sentence'.format(counter))
        return ret

    @staticmethod
    def _docid_from_guid(guid):
        tokens = guid[1:-1].replace('[', '').replace(']', ' ').split(' ')
        return '_'.join(tokens[0:-1])

    @staticmethod
    def _construct_sentence(doc, sent_text, sent_id):
        """
        :type doc: nlplingo.text.text_theory.Document
        :rtype: nlplingo.text.text_span.Sentence
        """
        offset = doc.text_length()
        if len(doc.sentences) > 0:
            offset += 1
        sentence_length_thus_far = 0
        tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""
        for i, token_string in enumerate(sent_text.split(' ')):
            start = offset + sentence_length_thus_far
            end = start + len(token_string)
            token = Token(IntPair(start, end), i, token_string, lemma=None, pos_tag=None)
            sentence_length_thus_far += len(token_string) + 1  # +1 for space
            tokens.append(token)
        sent = Sentence(doc.docid, IntPair(tokens[0].start_char_offset(), tokens[-1].end_char_offset()), sent_text,
                        tokens, len(doc.sentences))
        sent.sent_id = sent_id
        return sent

    def _use_sentence(self, sentence):
        if sentence.number_of_tokens() < 1:
            return False
        if sentence.number_of_tokens() >= self.max_sent_length:
            self.statistics['Skipping overly long sentence'] += 1
            return False
        return True

    def _doc_from_sentence_texts(self, docid, sentence_tuples):
        """Getting a list of tuples: (sentence-id, sentence-text)
        :type sentence_tuples: list[(str, str)]
        :rtype: nlplingo.text.text_theory.Document
        """
        doc = Document(docid)
        for sent_id, sent_text in sentence_tuples:
            sentence = WordPairGenerator._construct_sentence(doc, sent_text, sent_id)
            if self._use_sentence(sentence):
                doc.add_sentence(sentence)
        return doc

    @classmethod
    def _use_example(cls, example):
        """
        :type example: nlplingo.event.wordpair.WordPairExample
        """
        if example.wordpair_data[0] >= 0 and example.wordpair_data[1] >= 0:
            return True
        else:
            return False

    @classmethod
    def _generate_example(cls, example, tokens, max_sent_length, neighbor_dist, embedding_prefix):
        """
        :type example: nlplingo.event.wordpair.WordPairExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        anchor = example.anchor

        # we do a +1, becaause 'tokens' is actually: [None] + sentence.tokens
        anchor_token_indices = Struct(start=anchor.tokens[0].index_in_sentence + 1,
                                      end=anchor.tokens[-1].index_in_sentence + 1,
                                      head=anchor.head().index_in_sentence + 1)

        cls.assign_vector_data(tokens, example)
        cls.assign_position_data(anchor_token_indices, example, max_sent_length)  # position features
        window_text = cls.assign_lexical_data(anchor_token_indices, tokens, example, neighbor_dist)  # generate lexical features
        cls.assign_word_constraint_data(example, word_embeddings, embedding_prefix)

    @staticmethod
    def assign_word_constraint_data(example, embeddings, embedding_prefix):
        """
        :type example: nlplingo.event.wordpair.WordPairExample
        :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        vector_text, index, vector = embeddings.get_vector(example.word_text.lower(), embedding_prefix)
        if vector_text:
            example.wordpair_data[0] = index
        vector_text, index, vector = embeddings.get_vector(example.constraint_text.lower(), embedding_prefix)
        if vector_text:
            example.wordpair_data[1] = index

    @staticmethod
    def assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.wordpair.WordPairExample
        """
        for i, token in enumerate(tokens):
            if token and token.has_vector:
                example.vector_data[i] = token.vector_index

    @staticmethod
    def assign_position_data(anchor_token_indices, example, max_sent_length):
        """We capture positions of other words, relative to current word
        If the sentence is not padded with a None token at the front, then eg_index==token_index

        In that case, here is an example assuming max_sent_length==10 , and there are 4 tokens
        eg_index=0 , token_index=0    pos_data[0] = [ 0  1  2  3  4  5  6  7  8  9 ]  pos_index_data[0] = 0
        eg_index=1 , token_index=1    pos_data[1] = [-1  0  1  2  3  4  5  6  7  8 ]  pos_index_data[1] = 1
        eg_index=2 , token_index=2    pos_data[2] = [-2 -1  0  1  2  3  4  5  6  7 ]  pos_index_data[2] = 2
        eg_index=3 , token_index=3    pos_data[3] = [-3 -2 -1  0  1  2  3  4  5  6 ]  pos_index_data[3] = 3

        If the sentence is padded with a None token at the front, then eg_index==(token_index-1),
        and there are 5 tokens with tokens[0]==None

        eg_index=0 , token_index=1    pos_data[0] = [-1  0  1  2  3  4  5  6  7  8 ]  pos_index_data[0] = 1
        eg_index=1 , token_index=2    pos_data[1] = [-2 -1  0  1  2  3  4  5  6  7 ]  pos_index_data[1] = 2
        eg_index=2 , token_index=3    pos_data[2] = [-3 -2 -1  0  1  2  3  4  5  6 ]  pos_index_data[2] = 3
        eg_index=3 , token_index]4    pos_data[3] = [-4 -3 -2 -1  0  1  2  3  4  5 ]  pos_index_data[3] = 4

        * Finally, note that the code below adds self.gen.max_sent_length when assigning to pos_data.
        This is to avoid any negative values. For clarity of presentation, the above examples did not do this.

        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type example: event.event_trigger.EventTriggerExample
        """
        anchor_data = []
        for i in range(max_sent_length):
            if i < anchor_token_indices.start:
                anchor_data.append(i - anchor_token_indices.start + max_sent_length)
            elif anchor_token_indices.start <= i and i <= anchor_token_indices.end:
                anchor_data.append(0 + max_sent_length)
            else:
                anchor_data.append(i - anchor_token_indices.end + max_sent_length)
        example.pos_data[:] = anchor_data
        example.pos_index_data[0] = anchor_token_indices.head

    @staticmethod
    def assign_lexical_data(anchor_token_indices, tokens, example, neighbor_dist):
        """We want to capture [word-on-left , target-word , word-on-right]
        Use self.lex_data to capture context window, each word's embeddings or embedding index
        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.wordpair.WordPairExample
        :type max_sent_length: int
        :type neighbor_dist: int

        Returns:
            list[str]
        """
        # for lex_data, I want to capture: word-on-left target-word word-on-right
        # print('token_index=', token_index, ' eg_index=', eg_index)
        token_window = WordPairGenerator.get_token_window(tokens, anchor_token_indices, neighbor_dist)
        window_text = ['_'] * (2 * neighbor_dist + 1)
        for (i, token) in token_window:
            window_text[i] = token.text
            #example.lex_data[i] = token.vector_index
            # print('lex_data[', eg_index, ', ', i, '] = ', token.text)
            example.lex_data[i] = token.vector_index
        return window_text

    @staticmethod
    def window_indices(target_indices, window_size):
        """Generates a window of indices around target_index (token index within the sentence)

        :type target_indices: nlplingo.common.utils.Struct
        """
        indices = []
        indices.extend(range(target_indices.start - window_size, target_indices.start))
        indices.append(target_indices.head)
        indices.extend(range(target_indices.end + 1, target_indices.end + window_size + 1))
        return indices

    @staticmethod
    def get_token_window(tokens, token_indices, window_size):
        """
        :type token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        Returns:
            list[(int, nlplingo.text.text_span.Token)]

        As an example, let tokens = [None, XAgent, malware, linked, to, DNC, hackers, can, now, attack, Macs]
                                      0      1        2        3    4    5      6      7    8     9      10

        If we let target_index=1, window_size=1, so we want the window around 'XAgent'. This method returns:
        [(1, XAgent), (2, malware)]

        If we let target_index=2, this method returns:
        [(0, XAgent), (1, malware), (2, linked)]

        If we let target_index=10, this method returns:
        [(0, attack), (1, Macs)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(WordPairGenerator.window_indices(token_indices, window_size)):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            if token:
                ret.append((i, token))
        return ret

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.wordpair.WordPairExample]
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec'].append(example.vector_data)
            data_dict['lex_vec'].append(example.lex_data)
            data_dict['wordpair_vec'].append(example.wordpair_data)
            data_dict['pos_array'].append(example.pos_data)
            data_dict['pos_index'].append(example.pos_index_data)
            data_dict['label'].append(example.label)
        return data_dict

class WordPairExample(object):
    def __init__(self, anchor, sentence, params, label_string, word_text, constraint_text):
        """We are given a token, sentence as context, and event_type (present during training)
        :type anchor: nlplingo.text.text_span.Anchor
        :type sentence: nlplingo.text.text_span.Sentence
        :type params: nlplingo.common.parameters.Parameters
        :type label_string: str     # 'SAME' or 'DIFFERENT'
        """
        self.anchor = anchor
        self.sentence = sentence
        self.label_string = label_string
        self.label = pair_labels[label_string]
        self.word_text = word_text
        self.constraint_text = constraint_text
        self._allocate_arrays(params.get_int('max_sent_length'), params.get_int('cnn.neighbor_dist'),
                              params.get_int('embedding.none_token_index'), params.get_string('cnn.int_type'))

    def _allocate_arrays(self, max_sent_length, neighbor_dist, none_token_index, int_type):
        """Allocates feature vectors and matrices for examples from this sentence
        :type max_sent_length: int
        :type neighbor_dist: int
        :type none_token_index: int
        :type int_type: str
        """
        self.vector_data = none_token_index * np.ones(max_sent_length, dtype=int_type)
        self.lex_data = none_token_index * np.ones(2 * neighbor_dist + 1, dtype=int_type)
        self.wordpair_data = none_token_index * np.ones(2, dtype=int_type)  # word, constraint

        self.pos_data = np.zeros(max_sent_length, dtype=int_type)
        self.pos_index_data = np.zeros(1, dtype=int_type)

def generate_wordpair_data_feature(generator, docs, input_json, params):
    """
    :type generator: nlplingo.event.wordpair.WordPairGenerator
    :type docs: dict[str, nlplingo.text.text_theory.Document]
    """
    examples = generator.generate_examples(input_json, docs)
    """:type: list[nlplingo.event.wordpair.WordPairExample]"""

    data = generator.examples_to_data_dict(examples)
    if params.get_boolean('use_cnn'):
        data_list = [np.asarray(data['lex_vec']), np.asarray(data['wordpair_vec']), np.asarray(data['pos_array'])]
    else:
        data_list = [np.asarray(data['lex_vec']), np.asarray(data['wordpair_vec'])]
    label = np.asarray(data['label'])

    print('#wordpair-examples=%d' % (len(examples)))
    print('data word_vec.len=%d pos_array.len=%d label.len=%d' % (
        len(data['lex_vec']), len(data['pos_array']), len(data['label'])))

    for key in generator.statistics:
        print('{} = {}'.format(key, generator.statistics[key]))

    return (examples, data, data_list, label)


def evaluate_f1_binary(prediction, label, examples, class_label='OVERALL'):
    """
    :type examples: list[nlplingo.event.wordpair.WordPairExample]
    """
    num_correct = 0
    num_true = 0
    num_predict = 0
    for i in range(len(prediction)):
        pred = 0
        if prediction[i] >= 0.5:
            pred = 1
            num_predict += 1
            if label[i] == 1:
                num_correct += 1
        print('EXAMPLE docid={} word={} label={} pred={} prob={}'.format(examples[i].sentence.docid, examples[i].word_text, str(label[i]), str(pred), str(prediction[i])))

    for i in range(len(label)):
        if label[i] == 1:
            num_true += 1

    return F1Score(num_correct, num_true, num_predict, class_label)



def train(params, word_embeddings, generator):
    """
    :type generator: nlplingo.event.wordpair.WordPairGenerator
    """
    train_json = params.get_string('train.json')
    train_docs = dict()  # [docid] -> Document
    for doc in generator.generate_documents(train_json, word_embeddings):
        train_docs[doc.docid] = doc
    (train_examples, train_data, train_data_list, train_label) = generate_wordpair_data_feature(generator, train_docs,
                                                                                                train_json, params)

    event_domain = EventDomain([], [], [])
    wp_model = MaxPoolEmbeddedWordPairModel(params, word_embeddings, event_domain)
    wp_model.fit(train_data_list, train_label)

    print('==== Saving model ====')
    wp_model.save_keras_model(os.path.join(params.get_string('model_dir'), 'wordpair.hdf'))
    #with open(os.path.join(params.get_string('model_dir'), 'wordpair.pickle'), u'wb') as f:
    #    pickle.dump(wp_model, f)


def load_model(model_dir):
    model = keras_load_model(os.path.join(model_dir, 'wordpair.hdf'))
    return model
    #with open(os.path.join(model_dir, 'wordpair.pickle'), u'rb') as f:
    #    model = pickle.load(f)
    #    """:type: nlplingo.model.event_cnn.EventExtractionModel"""
    #    model.model_dir = model_dir
    #    return model


def test(params, word_embeddings, generator):
    """
    :type generator: nlplingo.event.wordpair.WordPairGenerator
    """
    print('==== Loading model ====')
    wp_model = load_model(params.get_string('model_dir'))

    test_jsons = params.get_list('test.json')
    for test_json in test_jsons:
        corpus_id = re.search(r'(.*)/data/(.*?)/', test_json).group(2)

        test_docs = dict()  # [docid] -> Document
        for doc in generator.generate_documents(test_json, word_embeddings):
            test_docs[doc.docid] = doc
        (test_examples, test_data, test_data_list, test_label) = generate_wordpair_data_feature(generator, test_docs,
                                                                                            test_json, params)
        predictions = wp_model.predict(test_data_list)
        score = evaluate_f1_binary(predictions, test_label, test_examples)

        print(score.to_string())
        with open(os.path.join(params.get_string('output_dir'), 'test_wordpair.' + corpus_id + '.score'), 'w') as f:
            f.write(score.to_string() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')   # train_wp, test_wp
    parser.add_argument('--params')
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    # load word embeddings
    word_embeddings = WordEmbedding(params, params.get_string('embedding.embedding_file'),
                                    params.get_int('embedding.vocab_size'), params.get_int('embedding.vector_size'))

    generator = WordPairGenerator(params, word_embeddings)

    if args.mode == 'train_wordpair':
        train(params, word_embeddings, generator)
    elif args.mode == 'test_wordpair':
        test(params, word_embeddings, generator)








