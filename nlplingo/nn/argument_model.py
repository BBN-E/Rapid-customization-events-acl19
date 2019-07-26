from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
import logging

from keras.layers import Flatten, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.constraints import maxnorm
from keras.layers import concatenate

from nlplingo.nn.event_model import EventExtractionModel
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
global keras_argument_model

logger = logging.getLogger(__name__)


class ArgumentModel(EventExtractionModel):
    def __init__(self, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.argument.feature.EventArgumentFeature
        """
        super(ArgumentModel, self).__init__(extractor_params, event_domain, embeddings)

        self.num_output = len(event_domain.event_roles)
        self.hyper_params = hyper_params
        self.features = features

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_argument_model

        logger.debug('- train_data_list={}'.format(train_data_list))
        logger.debug('- train_label={}'.format(train_label))

        none_label_index = self.event_domain.get_event_role_index('None')
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.hyper_params.positive_weight

        callbacks = None
        if self.hyper_params.early_stopping:
            callbacks = [early_stopping]

        history = keras_argument_model.fit(
            train_data_list,
            train_label,
            sample_weight=sample_weight,
            batch_size=self.hyper_params.batch_size,
            epochs=self.hyper_params.epoch,
            validation_data=(
                test_data_list,
                test_label
            ),
            callbacks=callbacks
        )

        return history

    def load_keras_model(self, filename=None):
        global keras_argument_model
        keras_argument_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_argument_model
        keras_argument_model.save(filename)

    def predict(self, test_data_list):
        global keras_argument_model

        try:
            pred_result = keras_argument_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'argument.hdf'))
            print('*** Loaded keras_argument_model ***')
            pred_result = keras_argument_model.predict(test_data_list)
        return pred_result


class CNNArgumentModel(ArgumentModel):
    def __init__(self, extractor_params, event_domain, embeddings, hyper_params, features):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        :type features: nlplingo.event.argument.feature.EventArgumentFeature
        """
        super(CNNArgumentModel, self).__init__(extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        global keras_argument_model

        model_input_dict = dict()
        outputs_to_merge1 = []

        if self.features.sentence_word_embedding:
            sentence_word_embedding_input = Input(shape=(self.hyper_params.max_sentence_length,),
                                                  name=u'sentence_word_embedding_input')
            outputs_to_merge1.append(Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                               weights=[self.word_embeddings],
                                               embeddings_initializer='glorot_uniform',
                                               trainable=self.hyper_params.train_embeddings)(
                sentence_word_embedding_input))
            model_input_dict[self.features.c_sentence_word_embedding] = sentence_word_embedding_input

        if self.features.trigger_word_position:
            trigger_word_position_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                                name=u'trigger_word_position')
            outputs_to_merge1.append(Embedding(2 * self.hyper_params.max_sentence_length,
                                               self.hyper_params.position_embedding_vector_length)(
                trigger_word_position_input))
            model_input_dict[self.features.c_trigger_word_position] = trigger_word_position_input

        if self.features.argument_word_position:
            argument_word_position_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                                 name=u'argument_word_position')
            outputs_to_merge1.append(Embedding(2 * self.hyper_params.max_sentence_length,
                                               self.hyper_params.position_embedding_vector_length)(
                argument_word_position_input))
            model_input_dict[self.features.c_argument_word_position] = argument_word_position_input

        if self.features.event_embeddings:
            event_embeddings_input = Input(shape=(self.hyper_params.max_sentence_length,), dtype=u'int32',
                                           name=u'event_embeddings')
            outputs_to_merge1.append(
                Embedding(self.num_event_types - 1, self.hyper_params.position_embedding_vector_length)(
                    event_embeddings_input))    # we do not embed the 'None' class
            model_input_dict[self.features.c_event_embeddings] = event_embeddings_input

        merged = concatenate(outputs_to_merge1, axis=-1)

        maxpools = []
        for filter_length in self.hyper_params.filter_lengths:
            conv = Convolution1D(self.hyper_params.number_of_feature_maps, filter_length, border_mode=u'valid',
                                 activation='relu')(merged)
            maxpools.append(GlobalMaxPooling1D()(conv))

        outputs_to_merge2 = []
        outputs_to_merge2.extend(maxpools)

        if self.features.trigger_argument_window:
            trigger_window = 2 * self.hyper_params.neighbor_distance + 1
            role_window = 2 * self.hyper_params.neighbor_distance + 1
            trigger_argument_window_input = Input(shape=(trigger_window + role_window,), dtype=u'int32',
                                                  name=u'trigger_argument_window')
            trigger_argument_window_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                                          weights=[self.word_embeddings],
                                                          embeddings_initializer='glorot_uniform',
                                                          trainable=self.hyper_params.train_embeddings)(
                trigger_argument_window_input)
            lex_flattened = Flatten()(trigger_argument_window_embedding)
            outputs_to_merge2.append(lex_flattened)
            model_input_dict[self.features.c_trigger_argument_window] = trigger_argument_window_input

        # Merge sentance and lexcial features
        merged_all = concatenate(outputs_to_merge2, axis=-1)

        # Dense MLP layer with dropout
        dropout = Dropout(self.hyper_params.dropout)(merged_all)
        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)

        assert set(self.features.feature_strings) == set(model_input_dict.keys())

        model_inputs = [model_input_dict[k] for k in self.features.feature_strings]

        keras_argument_model = Model(inputs=model_inputs, output=[out])

        keras_argument_model.compile(optimizer=self.optimizer, loss=u'categorical_crossentropy', metrics=[])
