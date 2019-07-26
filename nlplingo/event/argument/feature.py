
from itertools import chain

import numpy as np

from nlplingo.common.utils import Struct

class EventArgumentFeature(object):
    def __init__(self, features):
        """
        :type features: list[str]
        """
        self.feature_strings = features

        self.c_sentence_word_embedding = 'sentence_word_embedding'
        self.c_trigger_window = 'trigger_window'
        self.c_argument_window = 'argument_window'
        self.c_trigger_argument_window = 'trigger_argument_window'
        self.c_trigger_word_position = 'trigger_word_position'
        self.c_argument_word_position = 'argument_word_position'
        self.c_event_embeddings = 'event_embeddings'
        self.c_distance_between_trigger_argument = 'distance_between_trigger_argument'
        self.c_sentence_ner_type = 'sentence_ner_type'
        self.c_argument_ner_type = 'argument_ner_type'
        self.c_trigger_argument_relative_position = 'trigger_argument_relative_position'
        self.c_argument_unique_ner_type_in_sentence = 'argument_unique_ner_type_in_sentence'
        self.c_argument_is_nearest_ner_type = 'argument_is_nearest_ner_type'
        self.c_argument_nominal = 'argument_nominal'

        self.sentence_word_embedding = self.c_sentence_word_embedding in features
        self.trigger_argument_window = self.c_trigger_argument_window in features
        self.trigger_window = self.c_trigger_window in features
        self.argument_window = self.c_argument_window in features
        self.trigger_word_position = self.c_trigger_word_position in features
        self.argument_word_position = self.c_argument_word_position in features
        self.event_embeddings = self.c_event_embeddings in features
        self.distance_between_trigger_argument = self.c_distance_between_trigger_argument in features
        self.sentence_ner_type = self.c_sentence_ner_type in features
        self.argument_ner_type = self.c_argument_ner_type in features
        self.trigger_argument_relative_position = self.c_trigger_argument_relative_position in features
        self.argument_unique_ner_type_in_sentence = self.c_argument_unique_ner_type_in_sentence in features
        self.argument_is_nearest_ner_type = self.c_argument_is_nearest_ner_type in features
        self.argument_nominal = self.c_argument_nominal in features


class EventArgumentFeatureGenerator(object):
    def __init__(self, params):
        self.features = EventArgumentFeature(params['features'])
        """:type nlplingo.event.argument.feature.EventArgumentFeature"""


    def generate_example(self, example, tokens, hyper_params):
        """
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """

        anchor = example.anchor
        """:type: nlplingo.text.text_span.Anchor"""
        argument = example.argument
        """:type: nlplingo.text.text_span.EntityMention"""
        event_role = example.event_role

        # TODO need to handle multi-word arguments
        trigger_token_index = anchor.head().index_in_sentence
        # some triggers are multi-words, so we keep track of the start, end, and head token indices
        # trigger_token_indices = Struct(start=anchor.tokens[0].index_in_sentence,
        #                           end=anchor.tokens[-1].index_in_sentence, head=trigger_token_index)
        # we will just use the head token index, to be consistent with trigger training where we always use just the head for a trigger
        trigger_token_indices = Struct(start=trigger_token_index,
                                       end=trigger_token_index, head=trigger_token_index)

        arg_token_index = argument.head().index_in_sentence
        # some arguments are multi-words, so we keep track of the start, end, and head token indices
        arg_token_indices = Struct(start=argument.tokens[0].index_in_sentence,
                                   end=argument.tokens[-1].index_in_sentence, head=arg_token_index)

        # generate label data
        example.label[example.get_event_role_index()] = 1

        if self.features.trigger_window and self.features.argument_window:
            EventArgumentFeatureGenerator._assign_lexical_data(trigger_token_indices, arg_token_indices, tokens,
                                                               example, hyper_params.neighbor_distance,
                                                               self.features.argument_head)

        if self.features.event_embeddings:
            EventArgumentFeatureGenerator._assign_event_data(tokens, example)

        if self.features.sentence_word_embedding:
            EventArgumentFeatureGenerator._assign_vector_data(tokens, example)

        if self.features.trigger_word_position and self.features.argument_word_position:
            EventArgumentFeatureGenerator._assign_position_data(trigger_token_indices, arg_token_indices, example,
                                                            hyper_params.max_sentence_length)

        if self.features.distance_between_trigger_argument:
            EventArgumentFeatureGenerator._assign_lexical_distance(trigger_token_indices, arg_token_indices, example)

        if self.features.sentence_ner_type:
            EventArgumentFeatureGenerator._assign_ner_data(tokens, example)

        if self.features.argument_ner_type:
            EventArgumentFeatureGenerator._assign_role_ner_data(example, arg_token_index)

        # assign 0, 1, 2, 3 depending on whether trigger and argument: overlap, has single comma in betwee, arg before trigger, or trigger before arg
        if self.features.trigger_argument_relative_position:
            EventArgumentFeatureGenerator._assign_relative_pos(example, trigger_token_indices, arg_token_indices)

        if self.features.argument_unique_ner_type_in_sentence:
            EventArgumentFeatureGenerator._assign_unique_role_type(example, arg_token_index)

        if self.features.argument_is_nearest_ner_type:
            EventArgumentFeatureGenerator._assign_nearest_type(example, trigger_token_index, arg_token_index)

        if self.features.argument_nominal:
            EventArgumentFeatureGenerator._assign_common_name_lex(example, tokens, arg_token_index)


    @staticmethod
    def _window_indices(target_indices, window_size, use_head=True):
        """
        +1
        Generates a window of indices around target_index (token index within the sentence)
        :type target_indices: nlplingo.common.utils.Struct
        :type window_size: int
        :type use_head: bool
        """
        indices = []
        indices.extend(range(target_indices.start - window_size, target_indices.start))
        if use_head:
            indices.append(target_indices.head)
        indices.extend(range(target_indices.end + 1, target_indices.end + window_size + 1))
        return indices


    @classmethod
    def _get_token_windows(cls, tokens, window_size, trigger_token_indices, arg_token_indices, role_use_head):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type window_size: int
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type role_use_head: bool
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(chain(cls._window_indices(trigger_token_indices, window_size, use_head=True),
                                    cls._window_indices(arg_token_indices, window_size, use_head=role_use_head))):
            if w < 0 or w >= len(tokens):
                continue
            ret.append((i, tokens[w]))
        return ret


    @classmethod
    def _assign_lexical_distance(cls, trigger_token_indices, arg_token_indices, example):
        """
        +1
        Captures the distance between trigger and argument: 0: they overlap, 1: trigger and argument are neighboring tokens
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.util.Struct
        :type example: nlplingo.event.argument.example.EventArgumentExample
        """
        index_list = [
            arg_token_indices.start,
            trigger_token_indices.start,
            arg_token_indices.end,
            trigger_token_indices.end
        ]
        result = np.argsort(index_list, kind='mergesort')

        if list(result[1:3]) == [2, 1] or list(result[1:3]) == [3, 0]:
            example.lex_dist = (index_list[result[2]] - index_list[result[1]])
        else:
            example.lex_dist = 0  # They overlap so there is zero distance


    @classmethod
    def _assign_relative_pos(cls, example, trigger_token_indices, arg_token_indices):
        """
        +1
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        """
        def contains_comma(tokens, start, end):
            for tok in tokens[start:end]:
                if tok.text == ',':
                    return True
            return False

        ret_val = None
        index_list = [
            arg_token_indices.start,
            trigger_token_indices.start,
            arg_token_indices.end,
            trigger_token_indices.end
        ]
        result = np.argsort(index_list, kind='mergesort')

        # the intervals overlap
        # Since its a stable sort, if the first two elements after sorting are the arg_token_start and
        # trigger_token_start we can be assured that they are overlapping.

        if len({0, 1}.intersection(set(result[:2]))) == 2:
            ret_val = 0
        elif contains_comma(example.sentence.tokens, index_list[result[1]], index_list[result[2]]):
            ret_val = 1
        # arg is before the trigger
        elif arg_token_indices.end < trigger_token_indices.start:
            ret_val = 2
        # trigger is before the arg
        elif trigger_token_indices.end < arg_token_indices.start:
            ret_val = 3
        example.rel_pos = ret_val

    @classmethod
    def _assign_lexical_data_around_anchor(cls, anchor_token_window, var_to_assign, tokens):
        """
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        # get the local token windows around the trigger
        token_windows = cls._get_token_window(tokens, anchor_token_window)

        for (i, token) in token_windows:
            var_to_assign[i] = token.vector_index

    @classmethod
    def _get_token_window(cls, tokens, window_indices):
        """
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(window_indices):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            # if token:
            ret.append((i, token))
        return ret


    @classmethod
    def _assign_common_name_lex(cls, example, tokens, arg_token_index):
        """
        +1
        Look within the argument's coreference chain, and assign a nominal (common noun) embedding
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type arg_token_index: int
        """
        if example.argument.is_common_noun() or example.argument.entity is None:
            example.common_word_vec = tokens[arg_token_index].vector_index
        else:
            offset_to_mention = dict()
            for mention in example.argument.entity.mentions:
                if mention.head() is not None and mention.is_common_noun():
                    offset_to_mention[mention.start_char_offset()] = mention
                    #ret_val = mention.head().vector_index

            if len(offset_to_mention) > 0:
                offset, mention = sorted(offset_to_mention.items(), key=lambda s: s[0])[0]
                example.common_word_vec = mention.head().vector_index
            else:
                example.common_word_vec = tokens[arg_token_index].vector_index


    @classmethod
    def _assign_lexical_data(cls, trigger_token_indices, arg_token_indices, tokens, example, neighbor_dist, use_argument_head):
        """
        +1
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type neighbor_dist: int
        :type use_argument_head: bool
        """
        # get the local token windows around the trigger and argument
        token_windows = cls._get_token_windows(tokens, neighbor_dist, trigger_token_indices, arg_token_indices,
                                               use_argument_head)
        for (i, token) in token_windows:
            example.window_data[i] = token.vector_index     # local window around trigger and argument


    @staticmethod
    def _assign_event_data(tokens, example):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.argument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            example.event_data[i] = example.event_domain.get_event_type_index(example.anchor.label)


    @staticmethod
    def _assign_unique_role_type(example, arg_token_index):
        """
        +1
        checks whether the argument is the only entity-mention of that entity-type in the sentence
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type arg_token_index: int
        """
        ret_val = True
        arg_ne_type = example.ner_data[arg_token_index]
        for i, type in enumerate(example.ner_data):
            if i != arg_token_index and type == arg_ne_type:
                ret_val = False
                break
        example.is_unique_role_type = int(ret_val)


    @staticmethod
    def _assign_nearest_type(example, trigger_token_index, arg_token_index):
        """
        +1
        e.g. if the argument is of type PER. Then check whether there is another PER entity-mention nearer (than current argument) to the trigger
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type trigger_token_index: int
        :type arg_token_index: int
        """
        ret_val = True
        target_dist = abs(trigger_token_index - arg_token_index)
        arg_ne_type = example.ner_data[arg_token_index]
        for i, type in enumerate(example.ner_data):
            if i != arg_token_index and type == arg_ne_type:
                query_dist = abs(trigger_token_index - i)
                if query_dist < target_dist:
                    ret_val = False
                    break
        example.is_nearest_type = int(ret_val)


    @staticmethod
    def _assign_role_ner_data(example, arg_token_index):
        """
        +1
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :param arg_token_index: nlplingo.common.utils.Struct
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        example.role_ner[0] = example.event_domain.get_entity_type_index(token_ne_type[arg_token_index])


    @staticmethod
    def _assign_ner_data(tokens, example):
        """
        +1
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.argument.example.EventArgumentExample
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        assert len(token_ne_type) == len(tokens)
        for i, token in enumerate(tokens):
            example.ner_data[i] = example.event_domain.get_entity_type_index(token_ne_type[i])


    @staticmethod
    def _assign_vector_data(tokens, example):
        """
        +1
        Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.argument.example.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            example.vector_data[i] = token.vector_index


    # @staticmethod
    # def _assign_dep_data(example):
    #     """
    #     :type example: nlplingo.event.event_argument.EventArgumentExample
    #     """
    #     nearest_dep_child_distance = 99
    #     nearest_dep_child_token = None
    #     anchor_token_index = example.anchor.head().index_in_sentence
    #     for dep_r in example.anchor.head().child_dep_relations:
    #         if dep_r.dep_name == 'dobj':
    #             index = dep_r.child_token_index
    #             if abs(index - anchor_token_index) < nearest_dep_child_distance:
    #                 nearest_dep_child_distance = abs(index - anchor_token_index)
    #                 nearest_dep_child_token = example.sentence.tokens[dep_r.child_token_index]
    #
    #     if nearest_dep_child_token is not None:
    #         example.dep_data[0] = nearest_dep_child_token.vector_index
    #         example.anchor_obj = nearest_dep_child_token
    @staticmethod
    def _assign_dep_vector(example, tokens, trigger_index, argument_index):

        best_connect_dist = 10000
        for dep_rel in tokens[argument_index].dep_relations:
            if dep_rel.dep_direction == 'UP':

                modifier_text = dep_rel.dep_name
                target_text = tokens[dep_rel.connecting_token_index].text
                direction_modifier = 'I' if dep_rel.dep_direction == 'UP' else ''
                key = modifier_text + direction_modifier + '_' + target_text
                dep_data = tokens[argument_index].dep_rel_index_lookup.get(key, 0)

                if dep_data != 0:
                    connect_dist = abs(trigger_index - dep_rel.connecting_token_index)
                    if connect_dist < best_connect_dist:
                        example.arg_trigger_dep_data = dep_data
                        # This is important to the concept and wasnt there in initial testing JSF.
                        best_connect_dist = connect_dist

    @staticmethod
    def _assign_dep_data(example):
        """
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        anchor_head_token = example.anchor.head()
        anchor_token_index = anchor_head_token.index_in_sentence

        candidate_tokens = set()
        nearest_distance = 99
        nearest_token = None
        for srl in example.sentence.srls:
            if srl.predicate_token == anchor_head_token:
                if 'A1' in srl.roles:
                    for text_span in srl.roles['A1']:
                        # index = text_span.tokens[0].index_in_sentence
                        candidate_tokens.add(text_span.tokens[0])
                        # if abs(index - anchor_token_index) < nearest_distance:
                        #    nearest_distance = abs(index - anchor_token_index)
                        #    nearest_token = example.sentence.tokens[index]

        if nearest_token is None:
            for dep_r in example.anchor.head().child_dep_relations:
                if dep_r.dep_name == 'dobj':
                    index = dep_r.child_token_index
                    candidate_tokens.add(example.sentence.tokens[index])
                    # if abs(index - anchor_token_index) < nearest_distance:
                    #    nearest_distance = abs(index - anchor_token_index)
                    #    nearest_token = example.sentence.tokens[index]

        candidate_tokens_filtered = [t for t in candidate_tokens if t.pos_category() != 'PROPN']

        final_candidates = []
        if len(candidate_tokens_filtered) > 0:
            final_candidates = candidate_tokens_filtered
        else:
            final_candidates = candidate_tokens

        for t in final_candidates:
            index = t.index_in_sentence
            if abs(index - anchor_token_index) < nearest_distance:
                nearest_distance = abs(index - anchor_token_index)
                nearest_token = example.sentence.tokens[index]

        if nearest_token is not None:
            example.dep_data[0] = nearest_token.vector_index
            example.anchor_obj = nearest_token


    @staticmethod
    def _assign_token_texts(tokens, max_sent_length):
        """
        +1
        Lexical text of each token in the sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        """
        token_texts = ['_'] * max_sent_length
        for i, token in enumerate(tokens):
            token_texts[i] = u'{0}'.format(token.text)  # TODO want to use token.vector_text instead?
        return token_texts


    @staticmethod
    def _assign_position_data(trigger_token_indices, arg_token_indices, example, max_sent_length):
        """
        +1
        NOTE: you do not know whether index_pair[0] refers to the trigger_token_index or arg_token_index.
        Likewise for index_pair[1]. You only know that index_pair[0] < index_pair[1]

        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.argument.example.EventArgumentExample
        :type max_sent_length: int
        """
        # distance from trigger
        # example.pos_data[0, :] = [i - trigger_token_index + max_sent_length for i in range(max_sent_length)]
        trigger_data = []
        for i in range(max_sent_length):
            if i < trigger_token_indices.start:
                trigger_data.append(i - trigger_token_indices.start + max_sent_length)
            elif trigger_token_indices.start <= i and i <= trigger_token_indices.end:
                trigger_data.append(0 + max_sent_length)
            else:
                trigger_data.append(i - trigger_token_indices.end + max_sent_length)
        example.trigger_pos_data[:] = trigger_data

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
        example.argument_pos_data[:] = arg_data



