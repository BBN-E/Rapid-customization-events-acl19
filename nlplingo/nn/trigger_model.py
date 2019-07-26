from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import logging
import os

import keras
import numpy as np
from keras.layers import Flatten, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from nlplingo.nn.event_model import EventExtractionModel

global keras_trigger_model

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

logger = logging.getLogger(__name__)


class TriggerModel(EventExtractionModel):
    def __init__(self, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        """
        super(TriggerModel, self).__init__(extractor_params, event_domain, embeddings)
        self.number_of_entity_bio_types = len(event_domain.entity_bio_types)
        self.num_output = len(event_domain.event_types)
        self.model = None
        self.hyper_params = hyper_params
        self.features = features

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_trigger_model

        logger.debug('- train_data_list={}'.format(train_data_list))
        logger.debug('- train_label={}'.format(train_label))

        none_label_index = self.event_domain.get_event_type_index('None')
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.hyper_params.positive_weight

        callbacks = None
        if self.hyper_params.early_stopping:
            callbacks = [early_stopping]

        if len(test_label) > 0:
            history = keras_trigger_model.fit(
                train_data_list,
                train_label,
                sample_weight=sample_weight,
                batch_size=self.hyper_params.batch_size,
                nb_epoch=self.hyper_params.epoch,
                validation_data=(
                    test_data_list,
                    test_label
                ),
                callbacks=callbacks
            )
        else:
            history = keras_trigger_model.fit(
                train_data_list,
                train_label,
                sample_weight=sample_weight,
                batch_size=self.hyper_params.batch_size,
                nb_epoch=self.hyper_params.epoch,
                callbacks=callbacks
            )

        return history

    def load_keras_model(self, filename=None):
        global keras_trigger_model
        keras_trigger_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_trigger_model
        keras_trigger_model.save(filename)
        print(keras_trigger_model.summary())

    def predict(self, test_data_list):
        global keras_trigger_model

        try:
            pred_result = keras_trigger_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'trigger.hdf'))
            print('*** Loaded keras_trigger_model ***')
            pred_result = keras_trigger_model.predict(test_data_list)
        return pred_result


class CNNTriggerModel(TriggerModel):
    def __init__(self, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type extractor_params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.trigger.feature.EventTriggerFeature
        """
        super(CNNTriggerModel, self).__init__(extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        model_input_dict = dict()
        outputs_to_merge_1 = []

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings], trainable=self.hyper_params.train_embeddings,
                                    name=u'embedding_layer')
        # TODO : why was there a dropout=0.3 in the above Embedding layer?

        window_size = 2 * self.hyper_params.neighbor_distance + 1

        if self.features.sentence_word_embedding:
            sentence_word_embedding_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                                  name=u'sentence_word_embedding')
            outputs_to_merge_1.append(embedding_layer(sentence_word_embedding_input))
            model_input_dict[self.features.c_sentence_word_embedding] = sentence_word_embedding_input

        # For each word the pos_array_input defines the distance to the target work.
        # Embed each distance into an 'embedding_vec_length' dimensional vector space
        if self.features.trigger_word_position:
            trigger_word_position_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                                name=u'sentence_word_position')
            outputs_to_merge_1.append(Embedding(2 * self.hyper_params.max_sentence_length,
                                                self.hyper_params.position_embedding_vector_length)(
                trigger_word_position_input))
            model_input_dict[self.features.c_trigger_word_position] = trigger_word_position_input

        # Sentence feature input is the result of mergeing word vectors and embeddings
        if self.features.sentence_ner_type:
            sentence_ner_type_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                            name=u'sentence_entity_type')
            ner_embedding = Embedding(self.number_of_entity_bio_types,
                                      self.hyper_params.entity_embedding_vector_length)(sentence_ner_type_input)
            outputs_to_merge_1.append(ner_embedding)
            model_input_dict[self.features.c_sentence_ner_type] = sentence_ner_type_input

        merged = concatenate(outputs_to_merge_1, axis=-1)

        # Note: border_mode='same' to keep output the same width as the input
        maxpools = []
        for filter_length in self.hyper_params.filter_lengths:
            conv = Convolution1D(self.hyper_params.number_of_feature_maps, filter_length, border_mode=u'same',
                                 activation='relu')(merged)
            maxpools.append(GlobalMaxPooling1D()(conv))

        outputs_to_merge_2 = []
        outputs_to_merge_2.extend(maxpools)

        if self.features.trigger_window:
            trigger_window_input = Input(shape=(window_size,), dtype=u'int32', name=u'trigger_window')
            lex_words = embedding_layer(trigger_window_input)
            lex_flattened = Flatten()(lex_words)
            outputs_to_merge_2.append(lex_flattened)
            model_input_dict[self.features.c_trigger_window] = trigger_window_input

        merged_all = concatenate(outputs_to_merge_2)  # I used to use: merge(maxpools + [lex_flattened], mode=u'concat')

        # Dense MLP layer with dropout
        dropout = Dropout(self.hyper_params.dropout)(merged_all)
        out = Dense(self.num_output, activation=u'softmax')(dropout)

        model_inputs = [model_input_dict[k] for k in self.features.feature_strings]

        keras_trigger_model = Model(inputs=model_inputs, output=[out])
        keras_trigger_model.compile(optimizer=self.optimizer,
                                    loss=u'categorical_crossentropy',
                                    metrics=[])

        self.model = keras_trigger_model
