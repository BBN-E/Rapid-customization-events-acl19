from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
from keras.layers import Flatten, GlobalMaxPooling1D
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model

from nlplingo.model.my_keras import MyRange
from nlplingo.model.my_keras import MySelect
from nlplingo.model.event_cnn import EventExtractionModel

global keras_argument_model

class RoleModel(EventExtractionModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(RoleModel, self).__init__(params, event_domain, embeddings,
                                        batch_size=params.get_int('role.batch_size'),
                                        num_feature_maps=params.get_int('role.num_feature_maps'))
        self.num_output = len(event_domain.event_roles)
        self.positive_weight = params.get_float('role.positive_weight')
        self.entity_embedding_vec_length = params.get_int('role.entity_embedding_vec_length')  # entity embedding vector length
        self.epoch = params.get_int('role.epoch')
        if params is not None and params.has_key('role.use_event_embedding'):
            self.use_event_embedding = params.get_boolean('role.use_event_embedding')
        else:
            self.use_event_embedding = True

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_argument_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = self.event_domain.get_event_role_index('None')
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight
                
        #super(RoleModel, self).fit(train_label, train_data_list, test_label, test_data_list,
        #            sample_weight=sample_weight, max_epoch=self.epoch)
        history = keras_argument_model.fit(train_data_list, train_label,
                                          sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch,
                                          validation_data=(test_data_list, test_label))
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


class MaxPoolEmbeddedRoleModel(RoleModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedRoleModel, self).__init__(params, event_domain, embeddings)
        self.train_embedding = False
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        self.create_model()

    def create_model(self):
        global keras_argument_model

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        pos_array_input = Input(shape=(2, self.sent_length), dtype=u'int32', name=u'position_array')

        anchor_pos_array = MySelect(0)(pos_array_input)
        target_pos_array = MySelect(1)(pos_array_input)

        anchor_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(anchor_pos_array)
        target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(target_pos_array)

        # Each word is tagged with the SAME event number. Note sure why this event
        # number needs to be repeated, just following the description of the Chen et al
        # DMPooling paper. Embed event in vector space
        if self.use_event_embedding:
            event_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_array')
            event_embedding = Embedding(self.num_event_types, self.position_embedding_vec_len)(event_array_input)

        # ne_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'ne_array')
        # ne_embedding = Embedding(self.num_ne_types,
        #                             self.position_embedding_vec_len)(ne_input)

        # Input is matrix representing a sentence of self.sent_length tokens, where each token
        # is a vectors of length self.word_vec_length + 6
        # !!!! extra 6 tokens for the two tokens around trigger, and two tokens around role argument
        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        context_input = Input(shape=(self.sent_length + trigger_window + role_window,), name=u'word_vector')
        all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        context_word_input = MyRange(0, -(trigger_window + role_window))(all_words)

        # Sentence feature input is the result of mergeing word vectors and embeddings
        if self.use_event_embedding:
            merged = merge([context_word_input, anchor_embedding, target_embedding, event_embedding], mode=u'concat')
        else:
            merged = merge([context_word_input, anchor_embedding, target_embedding], mode=u'concat')

        # Note: border_mode='same' to keep output the same width as the input
        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')(merged)

        # Dynamically max pool 'conv' result into 3 max value
        maxpool = GlobalMaxPooling1D()(conv)

        # Input anchor and target words, plus +/- one context words
        # lex_input = Input(shape=(2 * self.num_lexical_tokens, self.word_vec_length), name=u'lex')
        lex_vector = MyRange(-(trigger_window + role_window), None)(all_words)

        # Lexical level feature
        lex_flattened = Flatten()(lex_vector)

        # Merge sentence and lexical features
        merged_all = merge([maxpool, lex_flattened], mode=u'concat')

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        # outputSize = numRoles
        out = Dense(self.num_output, activation=u'softmax')(dropout)

        if self.use_event_embedding:
            keras_argument_model = Model(input=[context_input, pos_array_input, event_array_input], output=[out])
        else:
            keras_argument_model = Model(input=[context_input, pos_array_input], output=[out])

        keras_argument_model.compile(optimizer=self.optimizer,
                                 loss=u'categorical_crossentropy',
                                 metrics=[])


class MaxPoolEmbeddedRoleNoTriggerModel(RoleModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedRoleNoTriggerModel, self).__init__(params, event_domain, embeddings)
        self.train_embedding = False
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        self.create_model()

    def create_model(self):
        global keras_argument_model

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array')

        target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(pos_array_input)

        # Input is matrix representing a sentence of self.sent_length tokens, where each token
        # is a vectors of length self.word_vec_length + 6
        # !!!! extra 3 tokens for the two tokens around role argument
        role_window = 2 * self.neighbor_dist + 1

        context_input = Input(shape=(self.sent_length,), name=u'word_vector')
        all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        target_input = Input(shape=(role_window,), name=u'target_vector')
        target_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                 weights=[self.word_embeddings], trainable=self.train_embedding)(target_input)

        merged = merge([all_words, target_embedding], mode=u'concat')

        # Note: border_mode='same' to keep output the same width as the input
        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')(merged)

        # Dynamically max pool 'conv' result into 3 max value
        maxpool = GlobalMaxPooling1D()(conv)

        # Lexical level feature
        lex_flattened = Flatten()(target_words)

        # Merge sentence and lexical features
        merged_all = merge([maxpool, lex_flattened], mode=u'concat')

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        # outputSize = numRoles
        out = Dense(self.num_output, activation=u'softmax')(dropout)

        keras_argument_model = Model(input=[context_input, pos_array_input, target_input], output=[out])

        keras_argument_model.compile(optimizer=self.optimizer,
                                     loss=u'categorical_crossentropy',
                                     metrics=[])
