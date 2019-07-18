from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from collections import defaultdict
import re
import codecs
import json

import numpy as np
from future.builtins import range

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import Struct
from nlplingo.common.io_utils import safeprint
from nlplingo.text.text_span import Anchor


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
            if pred_arg_max[i] != label_arg_max[i]:     # recall miss
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
            if pred_arg_max[i] != label_arg_max[i]:     # precision miss
                eg = examples[i]
                anchor_string = '_'.join(token.text.lower() for token in eg.anchor.tokens)
                misses['{}\t({})'.format(et, anchor_string)] += 1
    return misses


class EventKeywordList(object):
    def __init__(self, filepath):
        self.event_type_to_keywords = dict()
        self.keyword_to_event_types = defaultdict(set)
        self._read_keywords(filepath)

    def _read_keywords(self, filepath):
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        for data in datas:
            et_string = data['event_type']
            keywords = set()
            if 'keywords' in data:
                keywords.update(set(data['keywords']))
            if 'variants' in data:
                keywords.update(set(data['variants']))
            if 'hyponym_words' in data:
                keywords.update(set(data['hyponym_words']))
            if 'expanded_keywords' in data:
                keywords.update(set(data['expanded_keywords']))

            self.event_type_to_keywords[et_string] = set(kw.replace(' ', '_') for kw in keywords)
            for kw in keywords:
                self.keyword_to_event_types[kw].add(et_string)

    def get_event_types_for_tokens(self, tokens):
        """practically, we currently only deal with unigrams and bigrams
        :type tokens: list[nlplingo.text.text_span.Token]
        """
        ngrams = tokens[0:2]
        text = '_'.join(token.text for token in ngrams)
        text_with_pos_suffix = text + ngrams[-1].pos_suffix()

        event_types = set()
        if text_with_pos_suffix.lower() in self.keyword_to_event_types:
            event_types = self.keyword_to_event_types[text_with_pos_suffix.lower()]
        elif text.lower() in self.keyword_to_event_types:
            event_types = self.keyword_to_event_types[text.lower()]
        return event_types

    def print_statistics(self):
        # keywords which are associated with multiple event types
        for kw, ets in self.keyword_to_event_types.items():
            if len(ets) > 1:
                print('{} #ets={} {}'.format(kw, len(ets), ', '.join(ets)))


class EventTriggerGenerator(object):
    # we only accept tokens of the following part-of-speech categories as trigger candidates
    #trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
    trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])

    def __init__(self, event_domain, params, max_sentence_length, neighbor_distance, use_bio_index):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        """
        self.event_domain = event_domain
        #self.params = params
        self.max_sentence_length = max_sentence_length
        self.neighbor_distance = neighbor_distance
        self.use_bio_index = use_bio_index
        #self.do_trigger_pcnn = params.get_boolean('trigger.do_pcnn')
        # If the piecewise cnn were to be revised it should use the instantiation method used in extractor.py JSF
        self.do_trigger_pcnn = False

        self.negative_trigger_words = read_file_to_set(params['negative_trigger_words'])
        self.statistics = defaultdict(int)

        # for instance, some of the sentences we have not annotated in precursor, so those contain false negatives
        if 'sentence_ids_to_discard' in params:
            self.sentence_ids_to_discard = self._read_false_negatives(params['sentence_ids_to_discard'])
        else:
            self.sentence_ids_to_discard = dict()

        if 'trigger_candidate_span_file' in params:
            self.trigger_candidate_span = self._read_candidate_span_file(params['trigger_candidate_span_file'])
            """:type: defaultdict[str, set[IntPair]]"""
        else:
            self.trigger_candidate_span = None

        if 'np_span_file' in params:
            self.np_spans = self._read_candidate_span_file(params['np_span_file'])
            """:type: defaultdict[str, set[IntPair]]"""
        else:
            self.np_spans = None

        # if an anchor has this span, we will always generate a trigger example, event if we need to generate 'None' label
        if 'trigger.spans_to_generate_examples' in params:
            self.spans_to_generate_examples = self._read_candidate_span_file(params['trigger.spans_to_generate_examples'])
            """:type: defaultdict[str, set[IntPair]]"""
        else:
            self.spans_to_generate_examples = None

        self.restrict_none_examples_using_keywords = False
        if 'trigger.restrict_none_examples_using_keywords' in params:
            self.restrict_none_examples_using_keywords = params['trigger.restrict_none_examples_using_keywords']
            if self.restrict_none_examples_using_keywords:
                self.event_keyword_list = EventKeywordList(params['event_keywords'])

        if 'trigger.do_not_tag_as_none.file' in params:
            self.do_not_tag_as_none_dict = defaultdict(set)
            with codecs.open(params['trigger.do_not_tag_as_none.file'], 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    docid = tokens[0]
                    for w in tokens[1:]:
                        self.do_not_tag_as_none_dict[docid].add(w)
        else:
            self.do_not_tag_as_none_dict = None


    def _read_event_keywords(self, filepath):
        """ Read event keywords from a JSON file
        :rtype: dict[str, set[str]]
        """
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        ret = dict()
        reverse_map = defaultdict(set)
        for data in datas:
            et_string = data['event_type']
            keywords = set(data['keywords'] + data['variants'] + data['hyponym_words'])
            ret[et_string] = keywords
            for kw in keywords:
                reverse_map[kw].add(et_string)
        return ret, reverse_map

    def _read_candidate_span_file(self, filepath):
        ret = defaultdict(set)

        filepaths = []
        with open(filepath, 'r') as f:
            for line in f:
                filepaths.append(line.strip())
        
        for fp in filepaths:
            with codecs.open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    docid = tokens[0]
                    offset = IntPair(int(tokens[1]), int(tokens[2]))
                    ret[docid].add(offset)
        return ret

    def _read_false_negatives(self, filepath):
        ret = dict()
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line[line.index(' ')+1:]     # skip the first token/column
                docid_sentence_num = line[0:line.index(' ')]
                sentence_text = line[line.index(' ')+1:]
                ret[docid_sentence_num] = sentence_text
        return ret

    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        self.statistics.clear()

        self.negative_trigger = 0

        examples = []
        """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""

        for index, doc in enumerate(docs):
            #print('Generating trigger candidates from doc {}'.format(doc.docid))
            for sent in doc.sentences:
                doc_sent_id = '{}_{}'.format(sent.docid, sent.index)
                if doc_sent_id not in self.sentence_ids_to_discard:
                    examples.extend(self._generate_sentence(sent, doc.entity_mentions))
            if (index % 20) == 0:
                print('Generated examples from {} documents out of {}'.format(str(index + 1), str(len(docs))))

        for k, v in self.statistics.items():
            print('EventTriggerGenerator stats, {}:{}'.format(k, v))

        return examples

    def _generate_unigram_examples(self, sentence, tokens, entity_mentions):
        ret = []
        for token_index, token in enumerate(tokens):
            # TODO if current token is a trigger for multiple event types, event_type_index is only set to 1 event_type_index
            event_type = self.get_event_type_of_token(token, sentence)

            if event_type != 'None':
                self.statistics[token.pos_category()] += 1

            if not self.accept_tokens_as_candidate([token], event_type, entity_mentions, sentence.docid):
                continue

            self.statistics['number_candidate_trigger'] += 1
            if event_type != 'None':
                self.statistics['number_positive_trigger'] += 1

            anchor_candidate = Anchor('dummy-id', IntPair(token.start_char_offset(), token.end_char_offset()), token.text, event_type)
            anchor_candidate.with_tokens([token])
            example = EventTriggerExample(anchor_candidate, sentence, self.event_domain, self.max_sentence_length, self.neighbor_distance, event_type)
            self._generate_example(example, tokens, self.max_sentence_length, self.neighbor_distance, self.use_bio_index)
            ret.append(example)
            self.negative_trigger += 1

        return ret

    def _generate_bigram_examples(self, sentence, tokens):
        ret = []
        if self.np_spans is not None:
            doc_nps = self.np_spans[sentence.docid]     # set[IntPair]
            print('doc {} , len(doc_nps)={}, len(sentence.noun_phrases)={}'.format(sentence.docid, len(doc_nps), len(sentence.noun_phrases)))
            for np in sentence.noun_phrases:            # TextSpan
                for doc_np in doc_nps:
                    if np.start_char_offset() == doc_np.first and np.end_char_offset() == doc_np.second:
                        event_type = self.get_event_type_of_np(np, sentence)
                        self.statistics['number_candidate_trigger_np'] += 1
                        if event_type != 'None':
                            self.statistics['number_positive_trigger_np'] += 1
                        anchor_candidate = Anchor('dummy-id', IntPair(np.start_char_offset(), np.end_char_offset()), np.text, event_type)
                        anchor_candidate.with_tokens(np.tokens)
                        example = EventTriggerExample(anchor_candidate, sentence, self.event_domain, self.params, event_type)
                        self._generate_example(example, tokens, self.max_sent_length, self.neighbor_dist, self.self.params)
                        ret.append(example)
        return ret

    def _use_sentence(self, sentence):
        if sentence.number_of_tokens() < 1:
            return False
        if sentence.number_of_tokens() > self.max_sentence_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return False

        if self.trigger_candidate_span is not None:
            start = sentence.tokens[0].start_char_offset()
            end = sentence.tokens[0].end_char_offset()
            in_span = False
            for offset in self.trigger_candidate_span[sentence.docid]:
                if offset.first <= start and end <= offset.second:
                    in_span = True
                    break
            return in_span
        else:
            return True
        #return True

    def _generate_sentence(self, sentence, doc_entity_mentions):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :rtype: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        ret = []

        if not self._use_sentence(sentence):
            return ret

        #tokens = [None] + sentence.tokens
        #""":type: list[nlplingo.text.text_span.Token]"""

        self.statistics['number_event'] += len(sentence.events)
        self.statistics['number_trigger'] += len([event.anchors for event in sentence.events])

        ret.extend(self._generate_unigram_examples(sentence, sentence.tokens, sentence.entity_mentions))
        #ret.extend(self._generate_bigram_examples(sentence, tokens))

        return ret

    def accept_tokens_as_candidate(self, tokens, event_type, entity_mentions, docid):
        """Whether to reject a token as trigger candidate
        :type tokens: list[nlplingo.text.text_span.Token]
        :type entity_mentions: list[nlplingo.text.text_span.EntityMention]
        """
        # if self.trigger_candidate_span is not None:
        #     in_span = False
        #     for offset in self.trigger_candidate_span[docid]:
        #         if offset.first <= token.start_char_offset() and token.end_char_offset() <= offset.second:
        #             in_span = True
        #             break
        #     if not in_span:
        #         #print('Reject trigger because it does not fall within trigger_candidate_span')
        #         return False

        if self.spans_to_generate_examples is not None:
            for offset in self.spans_to_generate_examples[docid]:
                if offset.first == tokens[0].start_char_offset() and offset.second == tokens[-1].end_char_offset():
                    return True
                
        if tokens[-1].pos_category() not in self.trigger_pos_category:
            self.statistics['Reject trigger pos_category'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger pos_category: pos={} text={} {}'.format(tokens[-1].pos_tag, tokens[-1].text, event_type))
                self.statistics['Reject trigger pos_category TP'] += 1
            return False
        if "'" in tokens[-1].text:
            self.statistics['Reject trigger \''] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger \': text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger \' TP'] += 1
            return False
        if re.search('\d', tokens[-1].text):     # there's a digit
            self.statistics['Reject trigger digit'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger digit: text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger digit TP'] += 1
            return False
        if len(tokens[-1].text) < 2:
            self.statistics['Reject trigger len < 2'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger len < 2: text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger len < 2 TP'] += 1
            return False
        if tokens[-1].text.lower() in self.negative_trigger_words:
            self.statistics['Reject trigger negative-word'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger negative-word: text={} {}'.format(tokens[-1].text.lower(), event_type))
                self.statistics['Reject trigger negative-word TP'] += 1
            return False

        for em in entity_mentions:
            if em.head() is None:
                self.statistics['EM head is None'] += 1
                continue
            if em.head() == tokens[-1] and em.label != 'OTH':
                self.statistics['Reject trigger overlap-EM'] += 1
                if event_type != 'None':
                    safeprint(u'Reject trigger overlap-EM: docid={} text=({}) em-type={} {}'.format(docid, '_'.join(token.text.lower() for token in tokens), em.label, event_type))
                    self.statistics['Reject trigger overlap-EM TP'] += 1
                return False

        if self.restrict_none_examples_using_keywords and event_type == 'None':
            event_types = self.event_keyword_list.get_event_types_for_tokens(tokens)
            if len(event_types) > 0:
                self.statistics['Reject trigger as None since keyword for event-type'] += 1
                return False

        if self.do_not_tag_as_none_dict is not None and event_type == 'None' and docid in self.do_not_tag_as_none_dict:
            if tokens[-1].text.lower() in self.do_not_tag_as_none_dict[docid]:
                self.statistics['Reject trigger as None since keyword for event-type'] += 1
                return False

        if self.trigger_candidate_span is not None:
            if docid not in self.trigger_candidate_span:
                return False
            offsets = self.trigger_candidate_span[docid]
            for offset in offsets:
                if offset.first <= tokens[0].start_char_offset() and offset.second >= tokens[-1].end_char_offset():
                    return True
            return False

        return True


    @classmethod
    def _generate_example(cls, example, tokens, max_sentence_length, neighbor_distance, use_bio_index):
        """
        :type example: nlplingo.event.event_trigger.EventTriggerExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sentence_length: int
        :type neighbor_distance: int
        :type use_bio_index: bool
        """
        anchor = example.anchor
        event_domain = example.event_domain

        anchor_token_indices = Struct(start=anchor.tokens[0].index_in_sentence,
                                   end=anchor.tokens[-1].index_in_sentence, head=anchor.head().index_in_sentence)

        # TODO if current token is a trigger for multiple event types, event_type_index is only set to 1 event_type_index
        event_type_index = event_domain.get_event_type_index(example.event_type)

        example.label[event_type_index] = 1

        cls.assign_vector_data(tokens, example)
        if use_bio_index:
            cls._assign_entity_type_data(tokens, example)
        cls.assign_position_data(anchor_token_indices, example, max_sentence_length)  # position features
        window_text = cls.assign_lexical_data(anchor_token_indices, tokens, example, max_sentence_length, neighbor_distance)  # generate lexical features
        cls.assign_vector_data_array(anchor_token_indices, tokens, example, neighbor_distance)
        cls._assign_word_vector_data_dynamic(anchor_token_indices, max_sentence_length, example)
        cls._assign_word_vector_data_fb(anchor_token_indices, max_sentence_length, example)
        cls._assign_position_data_dynamic(anchor_token_indices, max_sentence_length, example)
        if use_bio_index:
            cls._assign_entity_type_data_dynamic(anchor_token_indices, max_sentence_length, example)

    @staticmethod
    def get_event_type_of_token(token, sent):
        """:type token: nlplingo.text.text_span.Token"""
        event_type = 'None'
        #print('target token, ', token.to_string())
        for event in sent.events:
            for anchor in event.anchors:
                #print('checking against anchor, ', anchor.to_string())
                if token.start_char_offset()==anchor.head().start_char_offset() and token.end_char_offset()==anchor.head().end_char_offset():
                    event_type = event.label
                    break
        return event_type

    @staticmethod
    def get_event_type_of_np(np, sent):
        """:type token: nlplingo.text.text_span.TextSpan"""
        event_type = 'None'
        for event in sent.events:
            for anchor in event.anchors:
                if anchor.start_char_offset() == np.start_char_offset() and anchor.end_char_offset() == np.end_char_offset():
                    event_type = event.label
                    break
        return event_type

    @staticmethod
    def assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_trigger.EventTriggerExample
        """
        for i, token in enumerate(tokens):
            #if token.has_vector:
            example.vector_data[i] = token.vector_index

    @staticmethod
    def assign_vector_data_array(anchor_token_indices, tokens, example, neighbor_dist):
        token_window = EventTriggerGenerator.get_token_window(tokens, anchor_token_indices, neighbor_dist)
        vectors = []
        for (i, token) in token_window:
            if token.word_vector is not None:  # It is None when the token is punct.
                vectors.append(token.word_vector)
        if len(vectors) > 0:
            example.vector_data_array = np.squeeze(np.stack(vectors))
        else:
            example.vector_data_array = None

    @staticmethod
    def _assign_word_vector_data_dynamic(anchor_token_indices, max_sent_length, example):
        for i in range(0, anchor_token_indices.end+1):
            example.vector_data_left[i] = example.vector_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            example.vector_data_right[i] = example.vector_data[i]

    @staticmethod
    def _assign_word_vector_data_fb(anchor_token_indices, max_sent_length, example):
        """

        :param anchor_token_indices: nlplingo.common.utils.Struct
        :param max_sent_length: int
        :param example: nlplingo.event.event_trigger.EventTriggerExample
        :return: None
        """
        context_length = anchor_token_indices.end + 1
        for i in range(0, anchor_token_indices.end+1):
            index = i + max_sent_length - context_length
            example.vector_data_forward[index] = example.vector_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            s_i = anchor_token_indices.start
            index = max_sent_length + s_i - i - 1
            example.vector_data_backward[index] = example.vector_data[i]


    @staticmethod
    def _assign_ner_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventTriggerExample
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        assert len(token_ne_type) == len(tokens)
        for i, token in enumerate(tokens):
            if token:
                example.ner_data[i] = example.event_domain.get_entity_type_index(token_ne_type[i])

    @staticmethod
    def _assign_entity_type_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventTriggerExample
        """
        token_ne_bio_type = example.sentence.get_ne_type_with_bio_per_token()
        assert len(token_ne_bio_type) == len(tokens)
        for i, token in enumerate(tokens):
            example.entity_type_data[i] = example.event_domain.get_entity_bio_type_index(token_ne_bio_type[i])

    @staticmethod
    def _assign_entity_type_data_dynamic(anchor_token_indices, max_sent_length, example):
        for i in range(anchor_token_indices.end + 1):
            example.entity_type_data_left[i] = example.entity_type_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            example.entity_type_data_right[i] = example.entity_type_data[i]

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
    def _assign_position_data_dynamic(anchor_token_indices, max_sent_length, example):
        for i in range(anchor_token_indices.end+1):
            example.pos_data_left[i] = example.pos_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            example.pos_data_right[i] = example.pos_data[i]

    @staticmethod
    def assign_lexical_data(anchor_token_indices, tokens, example, max_sent_length, neighbor_dist):
        """We want to capture [word-on-left , target-word , word-on-right]
        Use self.lex_data to capture context window, each word's embeddings or embedding index
        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_trigger.EventTriggerExample
        :type max_sent_length: int
        :type neighbor_dist: int

        Returns:
            list[str]
        """
        # for lex_data, I want to capture: word-on-left target-word word-on-right
        # print('token_index=', token_index, ' eg_index=', eg_index)
        token_window = EventTriggerGenerator.get_token_window(tokens, anchor_token_indices, neighbor_dist)
        window_text = ['_'] * (2 * neighbor_dist + 1)
        for (i, token) in token_window:
            window_text[i] = token.text
            #if token.has_vector:
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
        for i, w in enumerate(EventTriggerGenerator.window_indices(token_indices, window_size)):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            #if token:
            ret.append((i, token))
        return ret

    # @staticmethod
    # def get_token_window(tokens, target_index, window_size):
    #     """Gets the local tokens window
    #     :param tokens: list of tokens in the sentence
    #     :param target_index: indicates the target token. We want to get its surrounding tokens
    #     :param window_size: if this is e.g. 1, then we get its left and right neighbors
    #     :return: a list (max length 3) of tuples, giving the local window
    #
    #     As an example, let tokens = [None, XAgent, malware, linked, to, DNC, hackers, can, now, attack, Macs]
    #                                   0      1        2        3    4    5      6      7    8     9      10
    #
    #     If we let target_index=1, window_size=1, so we want the window around 'XAgent'. This method returns:
    #     [(1, XAgent), (2, malware)]
    #
    #     If we let target_index=2, this method returns:
    #     [(0, XAgent), (1, malware), (2, linked)]
    #
    #     If we let target_index=10, this method returns:
    #     [(0, attack), (1, Macs)]
    #
    #     :type tokens: list[nlplingo.text.text_span.Token]
    #     :type target_index: int
    #     :type window_size: int
    #     Returns:
    #         list[(int, nlplingo.text.text_span.Token)]
    #     """
    #     ret = []
    #     for i, w in enumerate(range(target_index - window_size, target_index + window_size + 1)):
    #         # I am now interested in token 'w'-th in tokens (which is the list of tokens of this sentence)
    #         if w < 0 or w >= len(tokens):  # falls outside the current tokens, so skip
    #             continue
    #         token = tokens[w]  # type(token) == spacy_wrapper.TokenWrapper
    #         if token:  # the 1st token might be a padded None token, so we check for this
    #             ret.append((i, token))
    #     return ret

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventTriggerExample
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec'].append(example.vector_data)
            data_dict['word_vec_left'].append(example.vector_data_left)
            data_dict['word_vec_right'].append(example.vector_data_right)
            data_dict['word_vec_forward'].append(example.vector_data_forward)
            data_dict['word_vec_backward'].append(example.vector_data_backward)
            data_dict['pos_array'].append(example.pos_data)
            data_dict['pos_array_left'].append(example.pos_data_left)
            data_dict['pos_array_right'].append(example.pos_data_right)
            data_dict['entity_type_array'].append(example.entity_type_data)
            data_dict['entity_type_array_left'].append(example.entity_type_data_left)
            data_dict['entity_type_array_right'].append(example.entity_type_data_right)
            data_dict['pos_index'].append(example.pos_index_data)
            data_dict['lex'].append(example.lex_data)
            data_dict['vector_data_array'].append(example.vector_data_array)
            data_dict['label'].append(example.label)
        return data_dict

class EventTriggerExample(object):
    def __init__(self, anchor, sentence, event_domain, max_sentence_length, neighbor_distance, event_type=None):
        """We are given a token, sentence as context, and event_type (present during training)
        :type anchor: nlplingo.text.text_span.Anchor
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type event_type: str
        """
        self.anchor = anchor
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_type = event_type
        self.score = 0
        self._allocate_arrays(max_sentence_length, neighbor_distance)

    def _allocate_arrays(self, max_sentence_length, neighbor_distance):
        """Allocates feature vectors and matrices for examples from this sentence
        :type max_sent_length: int
        :type neighbor_dist: int
        :type int_type: str
        """
        int_type = 'int32'
        num_labels = len(self.event_domain.event_types)

        self.label = np.zeros(num_labels, dtype=int_type)

        self.vector_data = np.zeros(max_sentence_length, dtype=int_type)
        # self.vector_data_left = none_token_index * np.ones(max_sent_length, dtype=int_type)
        self.vector_data_left = np.zeros(max_sentence_length, dtype=int_type)
        self.vector_data_right = np.zeros(max_sentence_length, dtype=int_type)
        self.vector_data_forward = np.zeros(max_sentence_length, dtype=int_type)
        self.vector_data_backward = np.zeros(max_sentence_length, dtype=int_type)

        self.lex_data = np.zeros(2 * neighbor_distance + 1, dtype=int_type)

        self.pos_data = np.zeros(max_sentence_length, dtype=int_type)
        self.pos_data_left = np.zeros(max_sentence_length, dtype=int_type)
        self.pos_data_right = np.zeros(max_sentence_length, dtype=int_type)

        self.entity_type_data = np.zeros(max_sentence_length, dtype=int_type)
        self.entity_type_data_left = np.zeros(max_sentence_length, dtype=int_type)
        self.entity_type_data_right = np.zeros(max_sentence_length, dtype=int_type)
        #self.entity_type_data = np.zeros((1))

        self.pos_index_data = np.zeros(1, dtype=int_type)
        self.all_text_output = []

        # maxtrix of -1 for each element
        #self.token_idx = -np.ones(max_sent_length, dtype=int_type)
        #self.ner_data = np.zeros(max_sent_length, dtype=int_type)

