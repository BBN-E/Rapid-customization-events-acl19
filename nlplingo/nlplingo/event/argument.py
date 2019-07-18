from itertools import chain
from collections import defaultdict

import numpy as np

from nlplingo.text.text_span import spans_overlap
from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import Struct
from nlplingo.event.event_trigger import EventTriggerGenerator


class ArgumentGenerator(object):
    verbosity = 0

    def __init__(self, event_domain, params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        """
        self.event_domain = event_domain
        self.params = params
        self.max_sent_length = params.get_int('max_sent_length')
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        self.role_use_head = params.get_boolean('role.use_head')
        self.statistics = defaultdict(int)

    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        for doc in docs:
            for sent in doc.sentences:
                examples.extend(self._generate_sentence(sent))

        for k, v in self.statistics.items():
            print('ArgumentGenerator stats, {}:{}'.format(k, v))

        return examples

    @staticmethod
    def get_event_role(entity_mention, events):
        """If the given (anchor, entity_mention) is found in the given events, return role label, else return 'None'
        :type entity_mention: nlplingo.text.text_span.EntityMention
        :type events: list[nlplingo.text.text_theory.Event]
        """
        ret = set()
        for event in events:
            role = event.get_role_for_entity_mention(entity_mention)
            if role != 'None':
                ret.add(role)
        return ret

    def _generate_sentence(self, sentence):
        """We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        """
        ret = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() >= self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        tokens = [None] + sentence.tokens

        event_types = defaultdict(list)
        for event in sentence.events:
            event_types[event.label].append(event)

        for event_type in event_types:
            for em in sentence.entity_mentions:
                roles = self.get_event_role(em, event_types[event_type])
                for role in roles:
                    self.statistics['#Event-Role {}'.format(role)] += 1

                if len(roles) > 0:
                    self.statistics['number_positive_argument'] += 1
                if len(roles) > 1:
                    self.statistics['number_positive_multi-label_argument'] += 1

                if len(roles) >= 1:
                    role = list(roles)[0]
                else:
                    role = 'None'
                example = ArgumentExample(event_type, em, sentence, self.event_domain, self.params, role)
                self._generate_example(example, tokens, self.max_sent_length, self.neighbor_dist)
                ret.append(example)
        return ret

    @classmethod
    def _generate_example(cls, example, tokens, max_sent_length, neighbor_dist):
        """
        :type example: nlplingo.event.event_argument.ArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        event_type = example.event_type
        argument = example.argument
        """:type: nlplingo.text.text_span.EntityMention"""
        event_role = example.event_role

        arg_token_index = argument.head().index_in_sentence + 1  # +1, to account for the [None] token
        # some arguments are multi-words, so we keep track of the start, end, and head token indices
        arg_token_indices = Struct(start=argument.tokens[0].index_in_sentence + 1,
                                   end=argument.tokens[-1].index_in_sentence + 1, head=arg_token_index)

        arg_role_index = example.get_event_role_index()

        # generate label data
        example.label[arg_role_index] = 1

        window_texts = cls._assign_lexical_data(arg_token_indices, tokens, example, neighbor_dist)

        cls._assign_vector_data(tokens, example)

        cls._assign_position_data(arg_token_indices, arg_token_index, example, max_sent_length)

        # cls._assign_ner_data(tokens, example)

        cls._assign_event_data(tokens, example)


    @staticmethod
    def _window_indices(target_indices, window_size, use_head=True):
        """Generates a window of indices around target_index (token index within the sentence)

        :type target_indices: nlplingo.common.utils.Struct
        """
        indices = []
        indices.extend(range(target_indices.start - window_size, target_indices.start))
        if use_head:
            indices.append(target_indices.head)
        indices.extend(range(target_indices.end + 1, target_indices.end + window_size + 1))
        return indices
        # return range(target_index - window_size, target_index + window_size + 1)

    @classmethod
    def _get_token_windows(cls, tokens, window_size, arg_token_indices, role_use_head):
        """
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        for i, w in enumerate(cls._window_indices(arg_token_indices, window_size, use_head=role_use_head)):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            if token:
                ret.append((i, token))
        return ret

    @classmethod
    def _assign_lexical_data(cls, arg_token_indices, tokens, example, neighbor_dist):
        """
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        # get the local token windows around the trigger and argument
        token_windows = cls._get_token_windows(tokens, neighbor_dist, arg_token_indices, example.role_use_head)

        if example.role_use_head:
            role_window = 2 * neighbor_dist + 1
        else:
            role_window = 2 * neighbor_dist

        window_texts = ['_'] * (role_window)

        for (i, token) in token_windows:
            if token.has_vector:
                example.lex_data[i] = token.vector_index
                #example.vector_data[i + max_sent_length] = token.vector_index
                window_texts[i] = token.text
            else:
                window_texts[i] = token.text

        return window_texts


    @staticmethod
    def _assign_ner_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        assert len(token_ne_type) + 1 == len(tokens)
        for i, token in enumerate(tokens):
            if token:
                example.ner_data[i] = example.event_domain.get_entity_type_index(token_ne_type[i - 1])

    @staticmethod
    def _assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            if token and token.has_vector:
                example.vector_data[i] = token.vector_index

    @staticmethod
    def _assign_position_data(arg_token_indices, arg_head_token_index, example, max_sent_length):
        """
        NOTE: you do not know whether index_pair[0] refers to the trigger_token_index or arg_token_index.
        Likewise for index_pair[1]. You only know that index_pair[0] < index_pair[1]

        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type max_sent_length: int
        """

        # distance from argument
        # example.pos_data[1, :] = [i - arg_token_index + max_sent_length for i in range(max_sent_length)]
        arg_data = []
        for i in range(max_sent_length):
            if i < arg_token_indices.start:
                arg_data.append(i - arg_token_indices.start + max_sent_length)
            elif arg_token_indices.start <= i and i <= arg_token_indices.end:
                arg_data.append(0 + max_sent_length)
            else:
                arg_data.append(i - arg_token_indices.end + max_sent_length)
        example.pos_data[:] = arg_data

        # for each example, keep track of the token index of trigger and argument
        example.pos_index_data[0] = arg_head_token_index

    @staticmethod
    def _assign_event_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.argument.ArgumentExample
        """
        for i, token in enumerate(tokens):
            if token:
                example.event_data[i] = example.event_domain.get_event_type_index(example.event_type)

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventArgumentExample
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec'].append(example.vector_data)
            data_dict['pos_array'].append(example.pos_data)
            data_dict['pos_index'].append(example.pos_index_data)
            data_dict['lex'].append(example.lex_data)
            data_dict['event_array'].append(example.event_data)
            data_dict['label'].append(example.label)

        return data_dict


class ArgumentExample(object):
    def __init__(self, event_type, argument, sentence, event_domain, params, event_role=None):
        """We are given an anchor, candidate argument, sentence as context, and a role label (absent in decoding)
        :type event_type: str
        :type argument: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        :type event_role: str
        """
        self.event_type = event_type
        self.argument = argument
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_role = event_role
        self.use_role = params.get_boolean('use_role')
        self.role_use_head = params.get_boolean('role.use_head')

        self._allocate_arrays(params.get_int('max_sent_length'), params.get_int('cnn.neighbor_dist'),
                              params.get_int('embedding.none_token_index'), params.get_string('cnn.int_type'),
                              # none_idx, params.get_string('cnn.int_type'),
                              params.get_boolean('role.use_head'))

    def get_event_role_index(self):
        return self.event_domain.get_event_role_index(self.event_role)

    def _allocate_arrays(self, max_sent_length, neighbor_dist, none_token_index, int_type, role_use_head):
        """
        :type max_sent_length: int
        :type neighbor_dist: int
        :type none_token_index: int
        :type int_type: str
        """
        num_labels = len(self.event_domain.event_roles)

        # self.label is a 2 dim matrix: [#instances , #event-roles], which I suspect will be 1-hot encoded
        self.label = np.zeros(num_labels, dtype=int_type)

        if role_use_head:
            role_window = 2 * neighbor_dist + 1
        else:
            role_window = 2 * neighbor_dist

        ## Allocate numpy array for data
        self.vector_data = np.zeros(max_sent_length, dtype=int_type)
        self.lex_data = np.zeros(role_window, dtype=int_type)
        self.vector_data[:] = none_token_index
        self.lex_data[:] = none_token_index

        self.pos_data = np.zeros(max_sent_length, dtype=int_type)
        # pos_index_data = [ #instances , 2 ]
        self.pos_index_data = np.zeros(1, dtype=int_type)
        # event_data = [ #instances , max_sent_length ]

        #self.all_text_output = []
        self.event_data = np.zeros(max_sent_length, dtype=int_type)
