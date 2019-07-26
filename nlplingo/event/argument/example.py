
import numpy as np


class EventArgumentExample(object):

    def __init__(self, anchor, argument, sentence, event_domain, params, extractor_params, features, hyper_params, event_role=None):
        """We are given an anchor, candidate argument, sentence as context, and a role label (absent in decoding)
        :type anchor: nlplingo.text.text_span.Anchor
        :type argument: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type features: nlplingo.event.argument.feature.EventArgumentFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type event_role: str
        """
        self.anchor = anchor
        self.argument = argument
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_role = event_role
        self.anchor_obj = None
        self.score = 0
        self._allocate_arrays(hyper_params, features, params['embeddings']['none_token_index'])

    def get_event_role_index(self):
        """
        +1
        """
        return self.event_domain.get_event_role_index(self.event_role)

    def _allocate_arrays(self, hyper_params, features, none_token_index):
        """
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features:  nlplingo.event.argument.feature.EventArgumentFeature
        """
        self.label = None
        self.vector_data = None         # sentence word embeddings
        self.lex_data = None            # local window around trigger and argument
        self.common_word_vec = None     # nominal representation (within coref chain) of argument head
        self.is_unique_role_type = None # if the argument the only entity mention of its entity-type in the sentence?
        self.is_nearest_type = None     # is there another entity mention (of the same entity type) nearest to the trigger?
        self.role_ner = None            # entity type of the argument
        self.trigger_pos_data = None    # relative position of sentence words, to the trigger
        self.argument_pos_data = None   # relative position of sentence words, to the argument
        self.lex_dist = None            # distance between trigger and argument
        self.rel_pos = None             # relative position between trigger and argument
        self.event_data = None          # event embeddings
        self.ner_data = None            # entity type of each word in the sentence

        int_type = 'int32'
        num_labels = len(self.event_domain.event_roles)

        self.label = np.zeros(num_labels, dtype=int_type)

        trigger_window = 2 * hyper_params.neighbor_distance + 1
        role_window = 2 * hyper_params.neighbor_distance + 1

        if features.sentence_word_embedding:
            self.vector_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)
            self.vector_data[:] = none_token_index

        if features.trigger_argument_window:
            self.lex_data = np.zeros(trigger_window + role_window, dtype=int_type)
            self.lex_data[:] = none_token_index

        if features.argument_nominal:
            self.common_word_vec = np.zeros(1, dtype=int_type)
            self.common_word_vec[:] = none_token_index

        if features.argument_unique_ner_type_in_sentence:
            self.is_unique_role_type = np.zeros(1, dtype=int_type)

        if features.argument_is_nearest_ner_type:
            self.is_nearest_type = np.zeros(1, dtype=int_type)

        if features.argument_ner_type:
            self.role_ner = np.zeros(1, dtype=int_type)
            self.role_ner[:] = self.event_domain.get_entity_type_index('None')

        if features.trigger_word_position and features.argument_word_position:
            self.trigger_pos_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)
            self.argument_pos_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)

        # distance between trigger and argument. (i) if they overlap, then lex_dist=0, (ii) if neighboring, then lex_dist=1
        if features.distance_between_trigger_argument:
            self.lex_dist = np.zeros(1, dtype=int_type)

        # relative position between trigger and argument. If rel_pos=0 then over-lapping. If rel_pos=1 then comma in between
        # If rel_pos=2 then argument before trigger. If rel_pos=3 then trigger before argument
        if features.trigger_argument_relative_position:
            self.rel_pos = np.zeros(1, dtype=int_type)

        if features.event_embeddings:
            self.event_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)
            self.event_data[:] = self.event_domain.get_event_type_index('None')

        self.all_text_output = []

        if features.sentence_ner_type:
            self.ner_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)
            self.ner_data[:] = self.event_domain.get_entity_type_index('None')


    def to_data_dict(self, features):
        """
        +1
        :type features:  nlplingo.event.argument.feature.EventArgumentFeature
        :rtype: dict[str:numpy.ndarray]
        """
        d = dict()

        if self.vector_data is not None:
            d[features.c_sentence_word_embedding] = self.vector_data
        if self.lex_data is not None:
            d[features.c_trigger_argument_window] = self.lex_data
        if self.common_word_vec is not None:
            d[features.c_argument_nominal] = self.common_word_vec
        if self.is_unique_role_type is not None:
            d[features.c_argument_unique_ner_type_in_sentence] = self.is_unique_role_type
        if self.is_nearest_type is not None:
            d[features.c_argument_is_nearest_ner_type] = self.is_nearest_type
        if self.role_ner is not None:
            d[features.c_argument_ner_type] = self.role_ner
        if self.trigger_pos_data is not None:
            d[features.c_trigger_word_position] = self.trigger_pos_data
        if self.argument_pos_data is not None:
            d[features.c_argument_word_position] = self.argument_pos_data
        if self.lex_dist is not None:
            d[features.c_distance_between_trigger_argument] = self.lex_dist
        if self.rel_pos is not None:
            d[features.c_trigger_argument_relative_position] = self.rel_pos
        if self.event_data is not None:
            d[features.c_event_embeddings] = self.event_data
        if self.ner_data is not None:
            d[features.c_sentence_ner_type] = self.ner_data

        d['label'] = self.label

        return d
