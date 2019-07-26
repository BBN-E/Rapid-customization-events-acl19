from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from collections import defaultdict
import re
import codecs
import json

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.common.utils import IntPair
from nlplingo.common.io_utils import safeprint
from nlplingo.text.text_span import Anchor

from nlplingo.event.trigger.example import EventTriggerExample
from nlplingo.event.trigger.feature import EventTriggerFeatureGenerator

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


class EventTriggerExampleGenerator(object):
    # we only accept tokens of the following part-of-speech categories as trigger candidates
    # trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
    trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        self.event_domain = event_domain
        self.params = params
        self.extractor_params = extractor_params
        self.hyper_params = hyper_params
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
            self.spans_to_generate_examples = self._read_candidate_span_file(
                params['trigger.spans_to_generate_examples'])
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
                line = line[line.index(' ') + 1:]  # skip the first token/column
                docid_sentence_num = line[0:line.index(' ')]
                sentence_text = line[line.index(' ') + 1:]
                ret[docid_sentence_num] = sentence_text
        return ret

    def generate(self, docs, feature_generator):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type feature_generator: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.trigger.trigger_example.EventTriggerExample]"""

        for index, doc in enumerate(docs):
            for sent in doc.sentences:
                doc_sent_id = '{}_{}'.format(sent.docid, sent.index)
                if doc_sent_id not in self.sentence_ids_to_discard:
                    examples.extend(self._generate_sentence(sent, feature_generator))
            if (index % 20) == 0:
                print('Generated examples from {} documents out of {}'.format(str(index + 1), str(len(docs))))

        for k, v in self.statistics.items():
            print('EventTriggerExampleGenerator stats, {}:{}'.format(k, v))

        return examples

    def _generate_unigram_examples(self, sentence, feature_generator, features, hyper_params):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator
        :type params: dict
        :type extractor_params: dict
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        ret = []
        for token_index, token in enumerate(sentence.tokens):
            # TODO if current token is a trigger for multiple event types, event_type_index is only set to 1 event_type_index
            event_type = EventTriggerFeatureGenerator.get_event_type_of_token(token, sentence)

            if not self.accept_tokens_as_candidate([token], event_type, sentence.entity_mentions, sentence.docid):
                continue

            self.statistics['number_candidate_trigger'] += 1
            if event_type != 'None':
                self.statistics[token.pos_category()] += 1
                self.statistics['number_positive_trigger'] += 1

            anchor_candidate = Anchor('dummy-id', IntPair(token.start_char_offset(), token.end_char_offset()),
                                      token.text, event_type)
            anchor_candidate.with_tokens([token])
            example = EventTriggerExample(anchor_candidate, sentence, self.event_domain, features, hyper_params, event_type)
            feature_generator.generate_example(example, sentence.tokens, hyper_params)
            ret.append(example)

        return ret

    def _generate_bigram_examples(self, sentence, params, extractor_params, features, hyper_params):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type params: dict
        :type extractor_params: dict
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        ret = []
        if self.np_spans is not None:
            doc_nps = self.np_spans[sentence.docid]  # set[IntPair]
            print('doc {} , len(doc_nps)={}, len(sentence.noun_phrases)={}'.format(sentence.docid, len(doc_nps),
                                                                                   len(sentence.noun_phrases)))
            for np in sentence.noun_phrases:  # TextSpan
                for doc_np in doc_nps:
                    if np.start_char_offset() == doc_np.first and np.end_char_offset() == doc_np.second:
                        event_type = self.get_event_type_of_np(np, sentence)
                        self.statistics['number_candidate_trigger_np'] += 1
                        if event_type != 'None':
                            self.statistics['number_positive_trigger_np'] += 1
                        anchor_candidate = Anchor('dummy-id', IntPair(np.start_char_offset(), np.end_char_offset()),
                                                  np.text, event_type)
                        anchor_candidate.with_tokens(np.tokens)
                        example = EventTriggerExample(anchor_candidate, sentence, self.event_domain, params, extractor_params, features, hyper_params, event_type)
                        EventTriggerFeatureGenerator.generate_example(example, sentence.tokens, hyper_params)
                        ret.append(example)
        return ret

    def _use_sentence(self, sentence):
        if sentence.number_of_tokens() < 1:
            return False
        if sentence.number_of_tokens() > self.hyper_params.max_sentence_length:
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

    def _generate_sentence(self, sentence, feature_generator):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.event.trigger.feature.EventTriggerFeatureGenerator
        :rtype: list[nlplingo.event.trigger.example.EventTriggerExample]
        """
        ret = []

        if not self._use_sentence(sentence):
            return ret

        self.statistics['number_event'] += len(sentence.events)
        self.statistics['number_trigger'] += len([event.anchors for event in sentence.events])

        ret.extend(self._generate_unigram_examples(sentence, feature_generator, feature_generator.features, self.hyper_params))
        #ret.extend(self._generate_bigram_examples(sentence, feature_generator, self.params, self.extractor_params, feature_generator.features, self.hyper_params))

        return ret

    def accept_tokens_as_candidate(self, tokens, event_type, entity_mentions, docid):
        """Whether to reject a token as trigger candidate
        :type tokens: list[nlplingo.text.text_span.Token]
        :type entity_mentions: list[nlplingo.text.text_span.EntityMention]
        """
        if self.spans_to_generate_examples is not None:
            for offset in self.spans_to_generate_examples[docid]:
                if offset.first == tokens[0].start_char_offset() and offset.second == tokens[-1].end_char_offset():
                    return True

        if tokens[-1].pos_category() not in self.trigger_pos_category:
            self.statistics['Reject trigger pos_category'] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger pos_category: pos={} text={} {}'.format(tokens[-1].pos_tag, tokens[-1].text,
                                                                                   event_type))
                self.statistics['Reject trigger pos_category TP'] += 1
            return False
        if "'" in tokens[-1].text:
            self.statistics['Reject trigger \''] += 1
            if event_type != 'None':
                safeprint(u'Reject trigger \': text={} {}'.format(tokens[-1].text, event_type))
                self.statistics['Reject trigger \' TP'] += 1
            return False
        if re.search('\d', tokens[-1].text):  # there's a digit
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
                    safeprint(u'Reject trigger overlap-EM: docid={} text=({}) em-type={} {}'.format(docid, '_'.join(
                        token.text.lower() for token in tokens), em.label, event_type))
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

    @staticmethod
    def examples_to_data_dict(examples, features):
        """
        :type examples: list[nlplingo.event.trigger.trigger_example.EventTriggerExample
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        """
        data_dict = defaultdict(list)
        for example in examples:
            example_data = example.to_data_dict(features)
            for k, v in example_data.items():
                data_dict[k].append(v)
        return data_dict
