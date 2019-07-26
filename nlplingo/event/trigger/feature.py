from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy as np
from future.builtins import range

from nlplingo.common.utils import Struct


class EventTriggerFeature(object):
    def __init__(self, features):
        """
        :type features: list[str]
        """
        self.feature_strings = features

        self.c_sentence_word_embedding = 'sentence_word_embedding'
        self.c_trigger_window = 'trigger_window'
        self.c_trigger_word_position = 'trigger_word_position'
        self.c_sentence_ner_type = 'sentence_ner_type'

        self.sentence_word_embedding = self.c_sentence_word_embedding in features
        self.trigger_window = self.c_trigger_window in features
        self.trigger_word_position = self.c_trigger_word_position in features
        self.sentence_ner_type = self.c_sentence_ner_type in features


class EventTriggerFeatureGenerator(object):
    # we only accept tokens of the following part-of-speech categories as trigger candidates
    # trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ', u'PROPN'])
    trigger_pos_category = set([u'NOUN', u'VERB', u'ADJ'])


    def __init__(self, params):
        self.features = EventTriggerFeature(params['features'])
        """:type nlplingo.event.trigger.feature.EventTriggerFeature"""


    def generate_example(self, example, tokens, hyper_params):
        """
        :type example: nlplingo.event.trigger.example.EventTriggerExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        anchor = example.anchor
        event_domain = example.event_domain

        anchor_token_indices = Struct(start=anchor.tokens[0].index_in_sentence,
                                      end=anchor.tokens[-1].index_in_sentence, head=anchor.head().index_in_sentence)

        event_type_index = event_domain.get_event_type_index(example.event_type)

        example.label[event_type_index] = 1

        if self.features.sentence_word_embedding:
            EventTriggerFeatureGenerator.assign_vector_data(tokens, example)

        if self.features.trigger_window:
            EventTriggerFeatureGenerator.assign_lexical_data(anchor_token_indices, tokens, example, hyper_params.neighbor_distance)

        if self.features.trigger_word_position:
            EventTriggerFeatureGenerator.assign_position_data(anchor_token_indices, example, hyper_params.max_sentence_length)

        if self.features.sentence_ner_type:
            EventTriggerFeatureGenerator._assign_entity_type_data(tokens, example)


    @staticmethod
    def get_event_type_of_token(token, sent):
        """:type token: nlplingo.text.text_span.Token"""
        event_type = 'None'
        # print('target token, ', token.to_string())
        for event in sent.events:
            for anchor in event.anchors:
                # print('checking against anchor, ', anchor.to_string())
                if token.start_char_offset() == anchor.head().start_char_offset() and token.end_char_offset() == anchor.head().end_char_offset():
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
        :type example: nlplingo.event.trigger.example.EventTriggerExample
        """
        for i, token in enumerate(tokens):
            example.sentence_data[i] = token.vector_index

    @staticmethod
    def assign_vector_data_array(anchor_token_indices, tokens, example, neighbor_dist):
        token_window = EventTriggerFeatureGenerator.get_token_window(tokens, anchor_token_indices, neighbor_dist)
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
        for i in range(0, anchor_token_indices.end + 1):
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
        for i in range(0, anchor_token_indices.end + 1):
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
        :type example: nlplingo.event.trigger.example.EventTriggerExample
        """
        anchor_data = []
        for i in range(max_sent_length):
            if i < anchor_token_indices.start:
                anchor_data.append(i - anchor_token_indices.start + max_sent_length)
            elif anchor_token_indices.start <= i and i <= anchor_token_indices.end:
                anchor_data.append(0 + max_sent_length)
            else:
                anchor_data.append(i - anchor_token_indices.end + max_sent_length)
        example.position_data[:] = anchor_data
        #example.pos_index_data[0] = anchor_token_indices.head

    @staticmethod
    def _assign_position_data_dynamic(anchor_token_indices, max_sent_length, example):
        for i in range(anchor_token_indices.end + 1):
            example.pos_data_left[i] = example.pos_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            example.pos_data_right[i] = example.pos_data[i]

    @staticmethod
    def assign_lexical_data(anchor_token_indices, tokens, example, neighbor_dist):
        """We want to capture [word-on-left , target-word , word-on-right]
        Use self.lex_data to capture context window, each word's embeddings or embedding index
        :type anchor_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.trigger.example.EventTriggerExample
        :type max_sent_length: int
        :type neighbor_dist: int

        Returns:
            list[str]
        """
        # for lex_data, I want to capture: word-on-left target-word word-on-right
        token_window = EventTriggerFeatureGenerator.get_token_window(tokens, anchor_token_indices, neighbor_dist)
        for (i, token) in token_window:
            example.window_data[i] = token.vector_index



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
        for i, w in enumerate(EventTriggerFeatureGenerator.window_indices(token_indices, window_size)):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            # if token:
            ret.append((i, token))
        return ret



