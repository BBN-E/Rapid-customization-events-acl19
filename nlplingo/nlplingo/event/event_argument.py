
from itertools import chain
from collections import defaultdict

import numpy as np

from nlplingo.text.text_span import spans_overlap
from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import Struct
from nlplingo.event.event_trigger import EventTriggerGenerator

class EventArgumentGenerator(object):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        """
        self.event_domain = event_domain
        self.params = params
        self.extractor_params = extractor_params
        self.max_sent_length = extractor_params['max_sentence_length']
        self.neighbor_dist = extractor_params['hyper-parameters']['neighbor_distance']
        self.role_use_head = extractor_params['model_flags']['use_head']
        self.do_dmcnn = extractor_params['model_flags'].get('do_dmcnn', False)
        self.statistics = defaultdict(int)

    def generate(self, docs, triggers=None):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type triggers: defaultdict(list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        for doc in docs:
            if triggers is not None:
                doc_triggers = triggers[doc.docid]
                """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""
                print('EventArgumentGenerator.generate(): doc.docid={}, len(doc_triggers)={}'.format(doc.docid, len(doc_triggers)))

                # organize the doc_triggers by sentence number
                sent_triggers = defaultdict(list)
                for trigger in doc_triggers:
                    sent_triggers[trigger.sentence.index].append(trigger)

                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent, trigger_egs=sent_triggers[sent.index]))
            else:
                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent))

        for k, v in self.statistics.items():
            print('EventArgumentGenerator stats, {}:{}'.format(k,v))

        return examples

    @staticmethod
    def get_event_role(anchor, entity_mention, events):
        """If the given (anchor, entity_mention) is found in the given events, return role label, else return 'None'
        :type anchor: nlplingo.text.text_span.Anchor
        :type entity_mention: nlplingo.text.text_span.EntityMention
        :type events: list[nlplingo.text.text_theory.Event]
        """

        for event in events:
            for a in event.anchors:
                if anchor.start_char_offset()==a.start_char_offset() and anchor.end_char_offset()==a.end_char_offset():
                    role = event.get_role_for_entity_mention(entity_mention)
                    if role != 'None':
                        return role
        return 'None'

    def _generate_sentence(self, sentence, trigger_egs=None):
        """We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        :type trigger_egs: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        # skip multi-token triggers, args that do not have embeddings, args that overlap with trigger
        ret = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        #tokens = [None] + sentence.tokens

        #sent_anchors = []
        if trigger_egs is not None:
            for trigger_index, eg in enumerate(trigger_egs):
                anchor_id = '{}-s{}-t{}'.format(sentence.docid, sentence.index, trigger_index)
                anchor = Anchor(anchor_id, IntPair(eg.anchor.start_char_offset(), eg.anchor.end_char_offset()), eg.anchor.text, eg.event_type)
                anchor.with_tokens(eg.anchor.tokens)
                #sent_anchors.append(a)
                for em in sentence.entity_mentions:
                    role = 'None'

                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                        self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                        ret.append(example)
        else:
            for event in sentence.events:
                for anchor in event.anchors:
                    if anchor.head().pos_category() in EventTriggerGenerator.trigger_pos_category:
                        for em in sentence.entity_mentions:
                            role = event.get_role_for_entity_mention(em)
                            self.statistics['#Event-Role {}'.format(role)] += 1
                            # if spans_overlap(anchor, em):
                            #     print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
                            # else:
                            #     if role != 'None':
                            #         self.statistics['number_positive_argument'] += 1
                            #     example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
                            #     self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                            #     ret.append(example)
                            if role != 'None':
                                self.statistics['number_positive_argument'] += 1
                            if em.coarse_label() in self.event_domain.entity_types.keys():
                                example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                                self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                                ret.append(example)

        # for anchor in sent_anchors:
        #     for em in sentence.entity_mentions:
        #         # TODO: this will be a problem if this anchor-word belongs to multiple events
        #         role = self.get_event_role(anchor, em, sentence.events)
        #         self.statistics['#Event-Role {}'.format(role)] += 1
        #
        #         # TODO
        #         if spans_overlap(anchor, em):
        #             if self.verbosity == 1:
        #                 print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
        #         else:
        #             if role != 'None':
        #                 self.statistics['number_positive_argument'] += 1
        #             example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
        #             self._generate_example(example, sentence.tokens, self.max_sent_length,
        #                                    self.neighbor_dist, self.do_dmcnn)
        #             ret.append(example)
        return ret

    @classmethod
    def _generate_example(cls, example, tokens, max_sent_length, neighbor_dist, do_dmcnn):
        """
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        :type neighbor_dist: int
        """

        anchor = example.anchor
        """:type: nlplingo.text.text_span.Anchor"""
        argument = example.argument
        """:type: nlplingo.text.text_span.EntityMention"""
        event_role = example.event_role

        # TODO need to handle multi-word arguments
        trigger_token_index = anchor.head().index_in_sentence
        # some triggers are multi-words, so we keep track of the start, end, and head token indices
        #trigger_token_indices = Struct(start=anchor.tokens[0].index_in_sentence,
        #                           end=anchor.tokens[-1].index_in_sentence, head=trigger_token_index)
        # we will just use the head token index, to be consistent with trigger training where we always use just the head for a trigger
        trigger_token_indices = Struct(start=trigger_token_index,
                                   end=trigger_token_index, head=trigger_token_index)

        arg_token_index = argument.head().index_in_sentence
        # some arguments are multi-words, so we keep track of the start, end, and head token indices
        arg_token_indices = Struct(start=argument.tokens[0].index_in_sentence,
                                   end=argument.tokens[-1].index_in_sentence, head=arg_token_index)
        # TODO : this needs to be reverted back to the above
        #arg_token_indices = Struct(start=arg_token_index,
        #                           end=arg_token_index, head=arg_token_index)

        arg_role_index = example.get_event_role_index()

        # generate label data
        example.label[arg_role_index] = 1

        #if trigger_token_index == arg_token_index:
        #    print('trigger_token_index={}, trigger={}, arg_token_index={}, arg={}'.format(trigger_token_index, tokens[trigger_token_index].text, arg_token_index, tokens[arg_token_index].text))
        #    print(argument.to_string())
        #    exit(0)

        # generate lexical features, assume single token triggers
        # self.lex_data = [ #instances , 2*(2*neighbor_dist+1) ]    # local-window around trigger and argument
        # also pads the window to end of self.vector_data
        cls._assign_lexical_data(trigger_token_indices, arg_token_indices, tokens, example, max_sent_length, neighbor_dist)

        # self.token_idx = [ #instances , self.gen.max_sent_length ]
        #cls._assign_token_idx(tokens, example)

        cls._assign_event_data(tokens, example)


        # self.vector_data = [ #instances , max_sent_length + 2*(2*neighbor_dist+1) ]
        cls._assign_vector_data(tokens, example)

        cls._assign_word_vector_data_fb(arg_token_indices, max_sent_length, example)

        # token_texts = [ self.gen.max_sent_length ]    # lexical text of each token in sentence
        #token_texts = cls._assign_token_texts(tokens, max_sent_length)

        # self.pos_data = [ #instances , 2, max_sent_length ]  # relative position of other words to trigger & arg
        # self.pos_index_data = [ #instances , 2 ]  # for each example, keep track of token index of trigger & arg
        index_pair = (min(trigger_token_index, arg_token_index), max(trigger_token_index, arg_token_index))
        cls._assign_position_data(trigger_token_indices, arg_token_indices, index_pair, example, max_sent_length)
        cls._assign_lexical_data_around_anchor(
            cls._window_indices(
                trigger_token_indices,
                neighbor_dist, use_head=True),
            example.lex_data_trigger,
            tokens
        )
        cls._assign_lexical_distance(example, trigger_token_indices, arg_token_indices)
        cls._assign_lexical_data_around_anchor(
            cls._window_indices(
                arg_token_indices,
                neighbor_dist, use_head=example.role_use_head
            ),
            example.lex_data_role,
            tokens
        )

        cls._assign_ner_data(tokens, example)
        #cls._assign_dep_data(example)
        has_dep_rel_data = any([len(x.dep_rel_index_lookup.keys()) > 0 for x in tokens])
        if has_dep_rel_data:
            cls._assign_dep_vector(example, tokens, trigger_token_index, arg_token_index)

        #cls._check_assertions(index_pair, tokens, trigger_token_index, arg_token_index,
        # argument.head(), example, max_sent_length)
        #cls._assign_sent_data(example)
        cls._assign_role_ner_data(example, arg_token_index)
        cls._assign_relative_pos(example, trigger_token_indices, arg_token_indices)
        cls._assign_unique_role_type(example, arg_token_index)
        cls._assign_nearest_type(example, trigger_token_index, arg_token_index)
        cls._assign_common_name_lex(example, tokens, arg_token_index)
        if do_dmcnn:
            cls._assign_event_data_dynamic(example, trigger_token_indices, arg_token_indices, len(tokens))
            cls._assign_vector_data_dynamic(example, trigger_token_indices, arg_token_indices, len(tokens))
            cls._assign_position_data_dynamic(example, trigger_token_indices, arg_token_indices, len(tokens))


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
        #return range(target_index - window_size, target_index + window_size + 1)

    @classmethod
    def _get_token_windows(cls, tokens, window_size, trigger_token_indices, arg_token_indices, role_use_head):
        """
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type tokens: list[nlplingo.text.text_span.Token]
        Returns:
            list[(int, nlplingo.text.text_span.Token)]
        """
        ret = []
        # chain just concatenates the 2 lists
        for i, w in enumerate(chain(cls._window_indices(trigger_token_indices, window_size, use_head=True),
                                    cls._window_indices(arg_token_indices, window_size, use_head=role_use_head))):
            if w < 0 or w >= len(tokens):
                continue
            token = tokens[w]
            #if token:
            ret.append((i, token))
        return ret

    @classmethod
    def _assign_lexical_distance(cls, example, trigger_token_indices, arg_token_indices):
        ret_val = None
        index_list = [
            arg_token_indices.start,
            trigger_token_indices.start,
            arg_token_indices.end,
            trigger_token_indices.end
        ]
        result = np.argsort(index_list, kind='mergesort')

        if list(result[1:3]) == [2, 1] or list(result[1:3]) == [3, 0]:
            ret_val = (index_list[result[2]] - index_list[result[1]])
        else:
            ret_val = 0  # They overlap so there is zero distance
        #print('Lexical distance {}'.format(ret_val))
        example.lex_dist = ret_val

    @classmethod
    def _assign_relative_pos(cls, example, trigger_token_indices, arg_token_indices):
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
        # get the local token windows around the trigger and argument
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
            #if token:
            ret.append((i, token))
        return ret

    @classmethod
    def _assign_common_name_lex(cls, example, tokens, arg_token_index):
        
        ret_val = tokens[arg_token_index].vector_index
        if example.argument.entity is not None:
            for mention in example.argument.entity.mentions:
                if mention.head() is not None:
                    if mention.is_common_noun():
                        ret_val = mention.head().vector_index
            example.common_word_vec = ret_val
        else:
            example.common_word_vec = 0

    @classmethod
    def _assign_lexical_data(cls, trigger_token_indices, arg_token_indices, tokens, example, max_sent_length, neighbor_dist):
        """
        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type max_sent_length: int
        :type neighbor_dist: int
        """
        # get the local token windows around the trigger and argument
        token_windows = cls._get_token_windows(tokens, neighbor_dist, trigger_token_indices, arg_token_indices, example.role_use_head)

        trigger_window = 2 * neighbor_dist + 1
        if example.role_use_head:
            role_window = 2 * neighbor_dist + 1
        else:
            role_window = 2 * neighbor_dist

        #window_texts = ['_'] * (trigger_window + role_window)

        for (i, token) in token_windows:
            #if token.has_vector:
            example.lex_data[i] = token.vector_index
            #window_texts[i] = token.text
            #else:
            #    window_texts[i] = token.text

        #return window_texts

    # @staticmethod
    # def _assign_token_idx(tokens, example):
    #     """
    #     :type tokens: list[nlplingo.text.text_span.Token]
    #     :type example: nlplingo.event.event_argument.EventArgumentExample
    #     """
    #     for i, token in enumerate(tokens):
    #         if token:
    #             example.token_idx[i] = token.spacy_token.i  # index of token within spacy document

    @staticmethod
    def _get_head_index(trigger_indices, argument_indices):
        left_index = min([trigger_indices.head, argument_indices.head])
        right_index = max([trigger_indices.head, argument_indices.head])
        return (left_index, right_index)

    @staticmethod
    def _assign_event_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            #if token:
            example.event_data[i] = example.event_domain.get_event_type_index(example.anchor.label)

    @staticmethod
    def _assign_event_data_dynamic(example, trigger_indices, argument_indices, sentence_length):
        index1, index2 = EventArgumentGenerator._get_head_index(trigger_indices, argument_indices)
        for i, index in enumerate(range(index1+1, index2)):
            example.event_data_middle[i] = example.event_data[index]
        for i in range(0, index1):
            example.event_data_left[i-index1] = example.event_data[i]
        for i in range(index2+1, sentence_length):
            example.event_data_right[i - index2] = example.event_data[i]

    @staticmethod
    def _assign_unique_role_type(example, arg_token_index):
        ret_val = True
        arg_ne_type = example.ner_data[arg_token_index]
        for i, type in enumerate(example.ner_data):
            if i != arg_token_index and type == arg_ne_type:
                ret_val = False
                break
        example.is_unique_role_type = int(ret_val)
        #print('{} {} {}'.format(i, type, )
    @staticmethod
    def _assign_nearest_type(example, trigger_token_index, arg_token_index):
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
        example.role_ner[0] = example.ner_data[arg_token_index]

    @staticmethod
    def _assign_ner_data(tokens, example):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        token_ne_type = example.sentence.get_ne_type_per_token()
        assert len(token_ne_type) == len(tokens)
        for i, token in enumerate(tokens):
            #if token:
            example.ner_data[i] = example.event_domain.get_entity_type_index(token_ne_type[i])

    @staticmethod
    def _assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        for i, token in enumerate(tokens):
            #if token.has_vector:
            example.vector_data[i] = token.vector_index

    @staticmethod
    def _assign_sent_data(example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        example.sent_data[0] = example.sentence.vector_index

    @staticmethod
    def _assign_word_vector_data_fb(anchor_token_indices, max_sent_length, example):
        context_length = anchor_token_indices.end + 1
        for i in range(0, anchor_token_indices.end+1):
            index = i + max_sent_length - context_length
            example.vector_data_forward[index] = example.vector_data[i]
        for i in range(anchor_token_indices.start, max_sent_length):
            s_i = anchor_token_indices.start
            index = max_sent_length + s_i - i - 1
            example.vector_data_backward[index] = example.vector_data[i]


    @staticmethod
    def _assign_vector_data_dynamic(example, trigger_indices, argument_indices, sentence_length):
        index1, index2 = EventArgumentGenerator._get_head_index(trigger_indices, argument_indices)
        for i, index in enumerate(range(index1 + 1, index2)):
            example.vector_data_middle[i] = example.vector_data[index]
        for i in range(0, index1):
            example.vector_data_left[i - index1] = example.vector_data[i]
        for i in range(index2+1, sentence_length):
            example.vector_data_right[i - index2] = example.vector_data[i]

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
                        #index = text_span.tokens[0].index_in_sentence
                        candidate_tokens.add(text_span.tokens[0])
                        #if abs(index - anchor_token_index) < nearest_distance:
                        #    nearest_distance = abs(index - anchor_token_index)
                        #    nearest_token = example.sentence.tokens[index]

        if nearest_token is None:
            for dep_r in example.anchor.head().child_dep_relations:
                if dep_r.dep_name == 'dobj':
                    index = dep_r.child_token_index
                    candidate_tokens.add(example.sentence.tokens[index])
                    #if abs(index - anchor_token_index) < nearest_distance:
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
        """Lexical text of each token in the sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        """
        token_texts = ['_'] * max_sent_length
        for i, token in enumerate(tokens):
            #if token:
            token_texts[i] = u'{0}'.format(token.text)  # TODO want to use token.vector_text instead?
        return token_texts

    @staticmethod
    def _assign_position_data(trigger_token_indices, arg_token_indices, index_pair, example, max_sent_length):
        """
        NOTE: you do not know whether index_pair[0] refers to the trigger_token_index or arg_token_index.
        Likewise for index_pair[1]. You only know that index_pair[0] < index_pair[1]

        :type trigger_token_indices: nlplingo.common.utils.Struct
        :type arg_token_indices: nlplingo.common.utils.Struct
        :type example: nlplingo.event.event_argument.EventArgumentExample
        :type max_sent_length: int
        """
        # distance from trigger
        #example.pos_data[0, :] = [i - trigger_token_index + max_sent_length for i in range(max_sent_length)]
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
        #example.pos_data[1, :] = [i - arg_token_index + max_sent_length for i in range(max_sent_length)]
        arg_data = []
        for i in range(max_sent_length):
            if i < arg_token_indices.start:
                arg_data.append(i - arg_token_indices.start + max_sent_length)
            elif arg_token_indices.start <= i and i <= arg_token_indices.end:
                arg_data.append(0 + max_sent_length)
            else:
                arg_data.append(i - arg_token_indices.end + max_sent_length)
        example.argument_pos_data[:] = arg_data
        
        # for each example, keep track of the token index of trigger and argument
        example.pos_index_data[0] = index_pair[0]
        example.pos_index_data[1] = index_pair[1]

    @staticmethod
    def _assign_position_data_dynamic(example, trigger_indices, argument_indices, sentence_length):
        index1, index2 = EventArgumentGenerator._get_head_index(trigger_indices, argument_indices)
        for i, index in enumerate(range(index1 + 1, index2)):
            example.trigger_pos_data_middle[i] = example.trigger_pos_data[index]
            example.argument_pos_data_middle[i] = example.argument_pos_data[index]
        for i in range(0, index1):
            example.trigger_pos_data_left[i - index1] = example.trigger_pos_data[i]
            example.argument_pos_data_left[i - index1] = example.argument_pos_data[i]
        for i in range(index2 + 1, sentence_length):
            example.trigger_pos_data_right[i - index2] = example.trigger_pos_data[i]
            example.argument_pos_data_right[i - index2] = example.argument_pos_data[i]

    @staticmethod
    def _check_assertions(index_pair, trunc_tokens, trunc_anchor_index, trunc_target_index, target_token, example, max_sent_length):
        """
        :type trunc_tokens: list[nlplingo.text.text_span.Token]
        :type target_token: nlplingo.text.text_span.Token
        :type example: nlplingo.event.event_argument.EventArgumentExample
        """
        assert (trunc_anchor_index == example.pos_index_data[index_pair.index(trunc_anchor_index)])
        assert (trunc_target_index == example.pos_index_data[1-index_pair.index(trunc_anchor_index)])
        assert (target_token == trunc_tokens[example.pos_index_data[1-index_pair.index(trunc_anchor_index)]])

        #if not (example.pos_index_data[0] < example.pos_index_data[1]):
        #    raise ValueError('0:{} 1:{}'.format(example.pos_index_data[0], example.pos_index_data[1]))
        assert (example.pos_data[0, trunc_anchor_index] == max_sent_length)
        assert (example.pos_data[1, trunc_target_index] == max_sent_length)
        assert np.all(example.pos_data[:, :] < 2*max_sent_length)
        assert np.all(example.pos_data[:, :] >= 0)
        offset = 1
        assert (np.min(example.pos_index_data[:]) >= offset)

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_argument.EventArgumentExample
        """
        data_dict = defaultdict(list)

        if self.do_dmcnn:
            print('examples_to_data_dict: do_dmcnn')
            for example in examples:
                data_dict['word_vec_left'].append(example.vector_data_left)
                data_dict['word_vec_middle'].append(example.vector_data_middle)
                data_dict['word_vec_right'].append(example.vector_data_right)

                data_dict['trigger_pos_array_left'].append(example.trigger_pos_data_left)
                data_dict['trigger_pos_array_middle'].append(example.trigger_pos_data_middle)
                data_dict['trigger_pos_array_right'].append(example.trigger_pos_data_right)

                data_dict['argument_pos_array_left'].append(example.argument_pos_data_left)
                data_dict['argument_pos_array_middle'].append(example.argument_pos_data_middle)
                data_dict['argument_pos_array_right'].append(example.argument_pos_data_right)

                data_dict['event_array_left'].append(example.event_data_left)
                data_dict['event_array_middle'].append(example.event_data_middle)
                data_dict['event_array_right'].append(example.event_data_right)

                data_dict['pos_index'].append(example.pos_index_data)
                data_dict['lex'].append(example.lex_data)
                data_dict['label'].append(example.label)

                data_dict['word_vec'].append(example.vector_data)
                data_dict['trigger_pos_array'].append(example.trigger_pos_data)
                data_dict['argument_pos_array'].append(example.argument_pos_data)
                data_dict['event_array'].append(example.event_data)
        else:
            print('examples_to_data_dict: !do_dmcnn')
            for example in examples:
                #data_dict['info'].append(example.info)
                data_dict['word_vec'].append(example.vector_data)
                data_dict['sent_vec'].append(example.sent_data)
                data_dict['trigger_pos_array'].append(example.trigger_pos_data)
                data_dict['argument_pos_array'].append(example.argument_pos_data)
                data_dict['lex_dist'].append(example.lex_dist)
                data_dict['event_array'].append(example.event_data)
                data_dict['pos_index'].append(example.pos_index_data)
                data_dict['lex'].append(example.lex_data)
                data_dict['label'].append(example.label)
                data_dict['word_vec_forward'].append(example.vector_data_forward)
                data_dict['word_vec_backward'].append(example.vector_data_backward)
                data_dict['lex_trigger'].append(example.lex_data_trigger)
                data_dict['lex_role'].append(example.lex_data_role)
                # for causal embeddings
                #data_dict['word_cvec'].append(example.cvector_data)
                #data_dict['clex'].append(example.clex_data)
                #data_dict['token_idx'].append(example.token_idx)
                data_dict['ne_array'].append(example.ner_data)
                data_dict['role_ner'].append(example.role_ner)
                data_dict['rel_pos'].append(example.rel_pos)
                data_dict['arg_trigger_dep'].append(example.arg_trigger_dep_data)
                data_dict['is_unique_role'].append(example.is_unique_role_type)
                data_dict['is_nearest_type'].append(example.is_nearest_type)
                data_dict['common_word'].append(example.common_word_vec)
        return data_dict


class EventArgumentExample(object):

    def __init__(self, anchor, argument, sentence, event_domain, params, extractor_params, event_role=None):
        """We are given an anchor, candidate argument, sentence as context, and a role label (absent in decoding)
        :type anchor: nlplingo.text.text_span.Anchor
        :type argument: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type event_role: str
        """
        self.anchor = anchor
        self.argument = argument
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_role = event_role
        self.role_use_head = extractor_params['model_flags']['use_head']
        self.do_dmcnn = extractor_params['model_flags'].get('do_dmcnn', False)
        self.anchor_obj = None
        self.score = 0
        """:type: nlplingo.text.text_span.Token"""

        self._allocate_arrays(
            extractor_params['max_sentence_length'],
            extractor_params['hyper-parameters']['neighbor_distance'],
            params['embeddings']['none_token_index'],
            extractor_params['int_type'],
            extractor_params['model_flags']['use_head']
        )

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

        # metadata info
        # i4: 32 bit integer , S40: 40 character string
        #self.info_dtype = np.dtype([(b'docid', b'S40'), (b'text_unit_no', b'i4'),
        #                           (b'trigger_token_no', b'i4'), (b'role_token_no', b'i4'),
        #                           (b'trigger_token_i', b'i4'), (b'role_token_i', b'i4')])
        #self.info = np.zeros(1, self.info_dtype)

        # self.label is a 2 dim matrix: [#instances , #event-roles], which I suspect will be 1-hot encoded
        self.label = np.zeros(num_labels, dtype=int_type)

        trigger_window = 2 * neighbor_dist + 1
        if role_use_head:
            role_window = 2 * neighbor_dist + 1
        else:
            role_window = 2 * neighbor_dist

        ## Allocate numpy array for data
        self.vector_data = np.zeros(max_sent_length, dtype=int_type)
        self.lex_data = np.zeros(trigger_window + role_window, dtype=int_type)
        self.lex_data_trigger = np.zeros(trigger_window, dtype=int_type)
        self.lex_data_trigger[:] = none_token_index
        self.lex_data_role = np.zeros(role_window, dtype=int_type)
        self.lex_data_role[:] = none_token_index
        self.common_word_vec = np.zeros(1, dtype=int_type)
        self.common_word_vec[:] = none_token_index
        self.sent_data = np.zeros(1, dtype=int_type)
        self.is_unique_role_type = np.zeros(1, dtype=int_type)
        self.is_nearest_type = np.zeros(1, dtype=int_type)
        self.vector_data[:] = none_token_index
        self.lex_data[:] = none_token_index
        self.dep_data = np.zeros(1, dtype=int_type) # dobj dependency
        self.dep_data[:] = none_token_index
        self.arg_trigger_dep_data = np.zeros(1, dtype=int_type) # dobj dependency
        self.arg_trigger_dep_data = 0
        self.role_ner = np.zeros(1, dtype=int_type)
        self.vector_data_forward = np.zeros(max_sent_length, dtype=int_type)
        self.vector_data_backward = np.zeros(max_sent_length, dtype=int_type)

        #self.cvector_data = np.zeros(max_sent_length, dtype=int_type)
        #self.clex_data = np.zeros(trigger_window + role_window, dtype=int_type)
        #self.cvector_data[:] = none_token_index
        #self.clex_data[:] = none_token_index

        # pos_data = [ #instances , 2 , max_sent_length ]
        #self.pos_data = np.zeros((2, max_sent_length), dtype=int_type)
        self.trigger_pos_data = np.zeros(max_sent_length, dtype=int_type)
        self.argument_pos_data = np.zeros(max_sent_length, dtype=int_type)
        self.lex_dist = np.zeros(1, dtype=int_type)

        # pos_index_data = [ #instances , 2 ]
        self.pos_index_data = np.zeros(2, dtype=int_type)
        self.rel_pos = np.zeros(1, dtype=int_type)

        # event_data = [ #instances , max_sent_length ]
        self.event_data = np.zeros(max_sent_length, dtype=int_type)
        self.event_data[:] = self.event_domain.get_event_type_index('None')

        self.all_text_output = []
        # token_idx = [ #instances , max_sent_length ] , maxtrix of -1 for each element
        #self.token_idx = -np.ones(max_sent_length, dtype=int_type)

        self.ner_data = np.zeros(max_sent_length, dtype=int_type)

        # dynamic
        if self.do_dmcnn:
            self.vector_data_left = np.zeros(max_sent_length, dtype=int_type)
            self.vector_data_left[:] = none_token_index
            self.vector_data_middle = np.zeros(max_sent_length, dtype=int_type)
            self.vector_data_middle[:] = none_token_index
            self.vector_data_right = np.zeros(max_sent_length, dtype=int_type)
            self.vector_data_right[:] = none_token_index

            self.trigger_pos_data_left = np.zeros(max_sent_length, dtype=int_type)
            self.trigger_pos_data_middle = np.zeros(max_sent_length, dtype=int_type)
            self.trigger_pos_data_right = np.zeros(max_sent_length, dtype=int_type)

            self.argument_pos_data_left = np.zeros(max_sent_length, dtype=int_type)
            self.argument_pos_data_middle = np.zeros(max_sent_length, dtype=int_type)
            self.argument_pos_data_right = np.zeros(max_sent_length, dtype=int_type)

            self.event_data_left = np.zeros(max_sent_length, dtype=int_type)
            self.event_data_middle = np.zeros(max_sent_length, dtype=int_type)
            self.event_data_right = np.zeros(max_sent_length, dtype=int_type)


class EventArgumentGeneratorBySentence(EventArgumentGenerator):
    def __init__(self, event_domain, params, extractor_params):
        super(EventArgumentGeneratorBySentence, self).__init__(event_domain, params, extractor_params)

    def generate(self, docs, triggers=None):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type triggers: defaultdict(list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        for doc in docs:
            if triggers is not None:
                doc_triggers = triggers[doc.docid]
                """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""
                print('EventArgumentGenerator.generate(): doc.docid={}, len(doc_triggers)={}'.format(doc.docid, len(doc_triggers)))

                # organize the doc_triggers by sentence number
                sent_triggers = defaultdict(list)
                for trigger in doc_triggers:
                    sent_triggers[trigger.sentence.index].append(trigger)

                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent, trigger_egs=sent_triggers[sent.index]))
            else:
                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent))

        for k, v in self.statistics.items():
            print('EventArgumentGenerator stats, {}:{}'.format(k,v))

        return examples

    def _generate_sentence(self, sentence, trigger_egs=None):
        """We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        :type trigger_egs: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        # skip multi-token triggers, args that do not have embeddings, args that overlap with trigger
        ret = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        #tokens = [None] + sentence.tokens

        #sent_anchors = []
        if trigger_egs is not None:
            for trigger_index, eg in enumerate(trigger_egs):
                anchor_id = '{}-s{}-t{}'.format(sentence.docid, sentence.index, trigger_index)
                anchor = Anchor(anchor_id, IntPair(eg.anchor.start_char_offset(), eg.anchor.end_char_offset()), eg.anchor.text, eg.event_type)
                anchor.with_tokens(eg.anchor.tokens)
                #sent_anchors.append(a)
                for em in sentence.entity_mentions:
                    role = 'None'

                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                        self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                        ret.append(example)
        else:
            event_label_to_event_anchor = defaultdict(list)
            for event in sentence.events:

                for anchor in event.anchors:
                    if anchor.head().pos_category() in EventTriggerGenerator.trigger_pos_category:
                        event_label_to_event_anchor[event.label].append((event, anchor))
            print('Sentence')
            for event_label, events_anchors in event_label_to_event_anchor.items():
                print('Event {}'.format(event_label))
                em_to_role = ['None'] * len(sentence.entity_mentions)
                for event, anchor in events_anchors:
                    for i, em in enumerate(sentence.entity_mentions):
                        role = event.get_role_for_entity_mention(em)
                        if role != 'None':
                            em_to_role[i] = role

                for i, em in enumerate(sentence.entity_mentions):
                    role = em_to_role[i]
                    anchor = events_anchors[0][1]
                    self.statistics['#Event-Role {}'.format(role)] += 1
                    print('{} {}'.format(i, role))
                    # if spans_overlap(anchor, em):
                    #     print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
                    # else:
                    #     if role != 'None':
                    #         self.statistics['number_positive_argument'] += 1
                    #     example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
                    #     self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                    #     ret.append(example)
                    if role != 'None':
                        self.statistics['number_positive_argument'] += 1
                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                        self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                        ret.append(example)
            print('End Sentence')
    # for anchor in sent_anchors:
        #     for em in sentence.entity_mentions:
        #         # TODO: this will be a problem if this anchor-word belongs to multiple events
        #         role = self.get_event_role(anchor, em, sentence.events)
        #         self.statistics['#Event-Role {}'.format(role)] += 1
        #
        #         # TODO
        #         if spans_overlap(anchor, em):
        #             if self.verbosity == 1:
        #                 print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
        #         else:
        #             if role != 'None':
        #                 self.statistics['number_positive_argument'] += 1
        #             example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
        #             self._generate_example(example, sentence.tokens, self.max_sent_length,
        #                                    self.neighbor_dist, self.do_dmcnn)
        #             ret.append(example)
        return ret

