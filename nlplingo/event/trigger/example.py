

import numpy as np


class EventTriggerExample(object):
    def __init__(self, anchor, sentence, event_domain, features, hyper_params, event_type=None):
        """We are given a token, sentence as context, and event_type (present during training)
        :type anchor: nlplingo.text.text_span.Anchor
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type event_type: str
        """
        self.anchor = anchor
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_type = event_type
        self.score = 0
        self._allocate_arrays(hyper_params, features)

    def _allocate_arrays(self, hyper_params, features):
        """Allocates feature vectors and matrices for examples from this sentence
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features:  nlplingo.event.trigger.feature.EventTriggerFeature
        """
        self.sentence_data = None
        self.window_data = None
        self.position_data = None
        self.entity_type_data = None

        int_type = 'int32'
        num_labels = len(self.event_domain.event_types)

        self.label = np.zeros(num_labels, dtype=int_type)

        if features.sentence_word_embedding:
            self.sentence_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)

        if features.trigger_window:
            self.window_data = np.zeros(2 * hyper_params.neighbor_distance + 1, dtype=int_type)

        if features.trigger_word_position:       # part of speech
            self.position_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)
        if features.sentence_ner_type:
            self.entity_type_data = np.zeros(hyper_params.max_sentence_length, dtype=int_type)


    def to_data_dict(self, features):
        """
        :type features:  nlplingo.event.trigger.feature.EventTriggerFeature
        :rtype: dict[str:numpy.ndarray]
        """
        d = dict()

        if self.sentence_data is not None:
            d[features.c_sentence_word_embedding] = self.sentence_data
        if self.window_data is not None:
            d[features.c_trigger_window] = self.window_data
        if self.position_data is not None:
            d[features.c_trigger_word_position] = self.position_data
        if self.entity_type_data is not None:
            d[features.c_sentence_ner_type] = self.entity_type_data
        d['label'] = self.label

        return d
