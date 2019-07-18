from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from collections import defaultdict

import numpy as np

class EventSentenceGenerator(object):

    def __init__(self, event_domain, params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        """
        self.event_domain = event_domain
        self.params = params
        self.max_sent_length = params.get_int('max_sent_length')
        self.statistics = defaultdict(int)

    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_sentence.EventSentenceExample]"""

        for doc in docs:
            for sent in doc.sentences:
                examples.extend(self._generate_sentence(sent))

        for k, v in self.statistics.items():
            print('EventSentenceGenerator stats, {}:{}'.format(k, v))

        return examples

    def _generate_sentence(self, sentence):
        """
        :type sentence: text.text_span.Sentence
        """
        ret = []
        """:type: list[nlplingo.event.event_sentence.EventSentenceExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() >= self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        tokens = sentence.tokens

        self.statistics['number_event'] += len(sentence.events)

        # get the set of event types for this sentence, or use 'None' if this sentence does not contain any positive event
        event_types = set()
        for event in sentence.events:
            event_types.add(event.label)
        if len(event_types) == 0:
            event_types.add('None')

        for event_type in event_types:
            example = EventSentenceExample(sentence, self.event_domain, self.params, event_type)
            self._generate_example(example, sentence.tokens)
            ret.append(example)
        return ret

    @classmethod
    def _generate_example(cls, example, tokens):
        """
        :type example: nlplingo.event.event_trigger.EventTriggerExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        """
        event_domain = example.event_domain

        # checks whether the current token is a trigger for multiple event types
        # this doesn't change the state of any variables, it just prints information
        # self.check_token_for_multiple_event_types(token, trigger_ids)

        # TODO if current token is a trigger for multiple event types, event_type_index is only set to 1 event_type_index
        event_type_index = event_domain.get_event_type_index(example.event_type)

        # self.label is a 2-dim matrix [#num_tokens, #event_types]
        example.label[event_type_index] = 1

        # self.vector_data = [ #instances , (max_sent_length + 2*neighbor_dist+1) ]
        # TODO why do we pad on 2*neighbor_dist+1 ??
        cls.assign_vector_data(tokens, example)

        # self.token_idx = [ #instances , self.gen.max_sent_length ]
        #cls.assign_token_idx(tokens, example)  # index of token within parent document

    @staticmethod
    def assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_sentence.EventSentenceExample
        """
        for i, token in enumerate(tokens):
            if token and token.has_vector:
                example.vector_data[i] = token.vector_index

    # @staticmethod
    # def assign_token_idx(tokens, example):
    #     """
    #     :type tokens: list[nlplingo.text.text_span.Token]
    #     :type example: nlplingo.event.event_sentence.EventSentenceExample
    #     """
    #     for i, token in enumerate(tokens):
    #         if token:
    #             example.token_idx[i] = token.spacy_token.i

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_sentence.EventSentenceExample]
        """
        data_dict = defaultdict(list)
        for example in examples:
            #data_dict['info'].append(example.info)
            data_dict['word_vec'].append(example.vector_data)
            data_dict['label'].append(example.label)
            data_dict['token_idx'].append(example.token_idx)
        return data_dict

class EventSentenceExample(object):

    def __init__(self, sentence, event_domain, params, event_type=None):
        """We are given a sentence as the event span, and event_type (present during training)
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        :type event_type: str
        """
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_type = event_type
        self._allocate_arrays(params.get_int('max_sent_length'), params.get_int('embedding.none_token_index'),
                              params.get_string('cnn.int_type'))

    def _allocate_arrays(self, max_sent_length, none_token_index, int_type):
        """Allocates feature vectors and matrices for examples from this sentence

        :type max_sent_length: int
        :type none_token_index: int
        :type int_type: str
        """
        num_labels = len(self.event_domain.event_types)

        # Allocate numpy array for label
        # self.label is a 2 dim matrix: [#instances X #event-types], which I suspect will be 1-hot encoded
        self.label = np.zeros(num_labels, dtype=int_type)

        # Allocate numpy array for data
        self.vector_data = none_token_index * np.ones(max_sent_length, dtype=int_type)

        self.all_text_output = []

        # maxtrix of -1 for each element
        #self.token_idx = -np.ones(max_sent_length, dtype=int_type)

