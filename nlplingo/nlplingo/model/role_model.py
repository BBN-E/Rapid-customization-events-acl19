from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import os
import keras
import numpy as np
from keras.layers import Flatten, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution1D
from keras.layers import concatenate
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.constraints import maxnorm
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import concatenate
from keras.layers import TimeDistributed

#from nlplingo.model.my_keras import MyRange
#from nlplingo.model.my_keras import MySelect
from nlplingo.model.event_cnn import EventExtractionModel
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
global keras_argument_model


class RoleModel(EventExtractionModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(RoleModel, self).__init__(extractor_params, event_domain, embeddings)
        hyper_params = extractor_params['hyper-parameters']
        model_flags = extractor_params['model_flags']

        self.num_output = len(event_domain.event_roles)
        self.neighbor_dist = hyper_params['neighbor_distance']
        self.position_embedding_vec_len = hyper_params['position_embedding_vector_length']
        self.num_feature_maps = hyper_params.get('number_of_feature_maps', 0)
        self.positive_weight = hyper_params['positive_weight']
        self.entity_embedding_vec_length = hyper_params['entity_embedding_vector_length']  # entity embedding vector length
        self.epoch = hyper_params['epoch']
        self.early_stopping = hyper_params.get('early_stopping', False)
        self.filter_length = hyper_params['cnn_filter_lengths'][0]
        self.batch_size = hyper_params['batch_size']
        self.dropout = hyper_params['dropout']
        self.use_event_embedding = model_flags.get('use_event_embedding', False)

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

        callbacks = None
        if self.early_stopping:
            callbacks = [early_stopping]

        history = keras_argument_model.fit(
            train_data_list,
            train_label,
            sample_weight=sample_weight,
            batch_size=self.batch_size,
            epochs=self.epoch,
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


class MaxPoolEmbeddedRoleModel(RoleModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedRoleModel, self).__init__(extractor_params, event_domain, embeddings)
        self.train_embedding = extractor_params['model_flags']['train_embeddings']
        if extractor_params['model_flags'].get('do_dmcnn', False):
            self.create_model_dmcnn()
        else:
            self.create_model()

    def create_model(self):
        global keras_argument_model

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        trigger_pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32',
                                   name=u'trigger_position_array')
        argument_pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32',
                                        name=u'argument_position_array')

        #anchor_pos_array = MySelect(0)(pos_array_input)
        #target_pos_array = MySelect(1)(pos_array_input)

        #anchor_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len, embeddings_initializer='glorot_uniform', trainable=True)(trigger_pos_array_input)
        #target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len, embeddings_initializer='glorot_uniform', trainable=True)(argument_pos_array_input)


        anchor_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(trigger_pos_array_input)
        target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(argument_pos_array_input)
        #position_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)
        #anchor_embedding = position_embedding(trigger_pos_array_input)
        #target_embedding = position_embedding(argument_pos_array_input)

        # Each word is tagged with the SAME event number. Note sure why this event
        # number needs to be repeated, just following the description of the Chen et al
        # DMPooling paper. Embed event in vector space

        event_embedding = None
        event_array_input = None
        if self.use_event_embedding:
            event_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_array')
            #event_embedding = Embedding(self.num_event_types-1, self.position_embedding_vec_len, embeddings_initializer='glorot_uniform', trainable=True)(event_array_input)
            event_embedding = Embedding(self.num_event_types - 1, self.position_embedding_vec_len)(event_array_input)

        #ne_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'ne_array')
        #ne_embedding = Embedding(self.num_ne_types,
        #                             self.position_embedding_vec_len)(ne_input)

        # Input is matrix representing a sentence of self.sent_length tokens, where each token
        # is a vectors of length self.word_vec_length + 6
        # !!!! extra 6 tokens for the two tokens around trigger, and two tokens around role argument
        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window + role_window,), dtype=u'int32', name=u'lex_vector')
        lex_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings],
                              #weights=[np.vstack((np.zeros((1, self.word_embeddings.shape[1])), self.word_embeddings))], mask_zero=True, input_length=(trigger_window+role_window),
                              embeddings_initializer='glorot_uniform',
                              trainable=self.train_embedding)(lex_input)

        context_input = Input(shape=(self.sent_length,), name=u'word_vector')
        all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings],
                              #weights=[np.vstack((np.zeros((1, self.word_embeddings.shape[1])), self.word_embeddings))], mask_zero=True, input_length=self.sent_length,
                              embeddings_initializer='glorot_uniform',
                              trainable=self.train_embedding)(context_input)


        #context_word_input = MyRange(0, -(trigger_window + role_window))(all_words)

        # Sentence feature input is the result of merging word vectors and embeddings

        outputs_to_merge = [all_words, anchor_embedding, target_embedding]

        if self.use_event_embedding:
            outputs_to_merge.append(event_embedding)

        merged = concatenate(outputs_to_merge)

        # Note: border_mode='same' to keep output the same width as the input
        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode='valid',
                             activation='relu'
                             )(merged)
        # Dynamially max pool 'conv' result into 3 max value
        maxpool = GlobalMaxPooling1D()(conv)
        #maxpool = MaxPooling1D(strides=1)(conv)
        #maxpool_flatten = Flatten()(maxpool)

        #conv2 = Convolution1D(self.num_feature_maps, 2, border_mode=u'same')(merged)
        # Dynamially max pool 'conv' result into 3 max value
        #maxpool2 = GlobalMaxPooling1D()(conv2)

        # Input anchor and target words, plus +/- one context words
        # lex_input = Input(shape=(2 * self.num_lexical_tokens, self.word_vec_length), name=u'lex')
        #lex_vector = MyRange(-(trigger_window + role_window), None)(all_words)

        # Lexical level feature
        lex_flattened = Flatten()(lex_words)

        # Merge sentance and lexcial features
        merged_all = concatenate([maxpool, lex_flattened])

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        # outputSize = numRoles
        #hidden = Dense(100, activation='relu', kernel_constraint=maxnorm(3))(
        #    merged_all)
        #out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(
        #    hidden)
        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)

        model_inputs = [
            context_input,
            trigger_pos_array_input,
            argument_pos_array_input,
            lex_input
        ]

        if self.use_event_embedding:
            model_inputs.append(event_array_input)

        keras_argument_model = Model(inputs=model_inputs, output=[out])

        keras_argument_model.compile(optimizer=self.optimizer,
                                 loss=u'categorical_crossentropy',
                                 metrics=[])

    def create_model_dmcnn(self):
        global keras_argument_model

        trigger_pos_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input')
        trigger_e = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                   embeddings_initializer='glorot_uniform')(trigger_pos_input)
        trigger_pos_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_left')
        trigger_e_left = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_left)
        trigger_pos_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_middle')
        trigger_e_middle = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_middle)
        trigger_pos_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_right')
        trigger_e_right = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_right)

        argument_pos_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input')
        argument_e = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(argument_pos_input)
        argument_pos_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_left')
        argument_e_left = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_left)
        argument_pos_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_middle')
        argument_e_middle = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_middle)
        argument_pos_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_right')
        argument_e_right = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_right)

        event_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input')
        event_e = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                 embeddings_initializer='glorot_uniform')(event_input)
        event_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_left')
        event_e_left = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_left)
        event_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_middle')
        event_e_middle = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_middle)
        event_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_right')
        event_e_right = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_right)

        word_input = Input(shape=(self.sent_length,), name=u'word_input')
        word_e = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                weights=[self.word_embeddings], trainable=self.train_embedding)(word_input)
        word_input_left = Input(shape=(self.sent_length,), name=u'word_input_left')
        word_e_left = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_left)
        word_input_middle = Input(shape=(self.sent_length,), name=u'word_input_middle')
        word_e_middle = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                           weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_middle)
        word_input_right = Input(shape=(self.sent_length,), name=u'word_input_right')
        word_e_right = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                           weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_right)

        merged = merge([word_e, trigger_e, argument_e, event_e], mode=u'concat')
        merged_left = merge([word_e_left, trigger_e_left, argument_e_left, event_e_left], mode=u'concat')
        merged_middle = merge([word_e_middle, trigger_e_middle, argument_e_middle, event_e_middle], mode=u'concat')
        merged_right = merge([word_e_right, trigger_e_right, argument_e_right, event_e_right], mode=u'concat')

        conv_all = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same',
                                 activation='relu')(merged)
        maxpool = GlobalMaxPooling1D()(conv_all)

        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same', activation='relu')

        #conv_left = Convolution1D(self.num_feature_maps, self.filter_length,
        # border_mode=u'same', activation='relu')(merged_left)
        #conv_middle = Convolution1D(self.num_feature_maps, self.filter_length,
        # border_mode=u'same', activation='relu')(merged_middle)
        #conv_right = Convolution1D(self.num_feature_maps, self.filter_length,
        # border_mode=u'same', activation='relu')(merged_right)

        conv_left = conv(merged_left)
        conv_middle = conv(merged_middle)
        conv_right = conv(merged_right)

        maxpool_left = GlobalMaxPooling1D()(conv_left)
        #maxpool_flatten_left = Flatten()(maxpool_left)

        maxpool_middle = GlobalMaxPooling1D()(conv_middle)
        #maxpool_flatten_middle = Flatten()(maxpool_middle)

        maxpool_right = GlobalMaxPooling1D()(conv_right)
        #maxpool_flatten_right = Flatten()(maxpool_right)

        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window + role_window,), dtype=u'int32', name=u'lex_input')
        lex_e = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(lex_input)
        lex_flattened = Flatten()(lex_e)

        merged_all = merge([maxpool, maxpool_left, maxpool_middle, maxpool_right, lex_flattened],
                           mode=u'concat')


        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)


        keras_argument_model = Model(input=[word_input, word_input_left, word_input_middle,
                                            word_input_right,
                                            trigger_pos_input, trigger_pos_input_left,
                                            trigger_pos_input_middle, trigger_pos_input_right,
                                            argument_pos_input, argument_pos_input_left,
                                            argument_pos_input_middle, argument_pos_input_right,
                                            event_input, event_input_left, event_input_middle,
                                            event_input_right,
                                            lex_input], output=[out])

        keras_argument_model.compile(optimizer=self.optimizer, loss=u'categorical_crossentropy', metrics=[])


class MaxPoolEmbeddedRoleNoTriggerModel(RoleModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedRoleNoTriggerModel, self).__init__(extractor_params, event_domain, embeddings)
        self.train_embedding = extractor_params['model_flags']['train_embeddings']
        self.neighbor_dist = extractor_params['hyper-parameters']['neighbor_distance']
        self.create_model()

    def create_model(self):
        global keras_argument_model

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array')

        target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(pos_array_input)

        event_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_array')
        event_embedding = Embedding(self.num_event_types, self.position_embedding_vec_len)(event_array_input)

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

        merged = merge([all_words, target_embedding, event_embedding], mode=u'concat')

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

        keras_argument_model = Model(input=[context_input, pos_array_input, target_input, event_array_input], output=[out])

        keras_argument_model.compile(optimizer=self.optimizer,
                                     loss=u'categorical_crossentropy',
                                     metrics=[])

## BidirectionalRoleModel


class BidirectionalRoleModel(EventExtractionModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(BidirectionalRoleModel, self).__init__(
            extractor_params,
            event_domain,
            embeddings
        )
        self.train_embedding = extractor_params['model_flags']['train_embeddings']

        hyper_params = extractor_params['hyper-parameters']
        model_flags = extractor_params['model_flags']

        self.num_output = len(event_domain.event_roles)
        self.positive_weight = hyper_params['positive_weight']
        self.epoch = hyper_params['epoch']
        #self.filter_length = hyper_params['cnn_filter_lengths'][0]
        self.batch_size = hyper_params['batch_size']
        self.dropout = hyper_params['dropout']
        self.n_lstm_cells = hyper_params['n_lstm_cells']
        self.create_model()

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

        # super(RoleModel, self).fit(train_label, train_data_list, test_label, test_data_list,
        #            sample_weight=sample_weight, max_epoch=self.epoch)
        history = keras_argument_model.fit(train_data_list, train_label,
                                           sample_weight=sample_weight, batch_size=self.batch_size, epochs=self.epoch,
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

    def create_model(self):
        global keras_argument_model

        '''
        context_input = Input(shape=(self.sent_length,), name=u'word_vector')
        all_words = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding
        )(context_input)


        bidirectional = Bidirectional(LSTM(20, return_sequences=True), merge_mode='concat')(all_words)
        out = TimeDistributed(Dense(self.num_output, activation='sigmoid'))(bidirectional)

        keras_argument_model = Model(input=[context_input], output=[out])

        keras_argument_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )
        '''
        input_forward = Input(shape=(self.sent_length,), name=u'word_vector_forward')
        input_backward = Input(shape=(self.sent_length,), name=u'word_vector_backward')
        forward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_forward)
        backward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_backward)

        forward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1)
        )(forward_emb)
        backward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1)
        )(backward_emb)

        merged = concatenate([forward_lstm, backward_lstm])

        out = Dense(self.num_output, activation='sigmoid')(merged)

        keras_argument_model = Model(
            inputs=[input_forward, input_backward],
            output=[out]
        )

        keras_argument_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )


class BiLSTMMaxPoolEmbeddedRoleModel(RoleModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(BiLSTMMaxPoolEmbeddedRoleModel, self).__init__(extractor_params, event_domain, embeddings)
        self.train_embedding = extractor_params['model_flags']['train_embeddings']

        hyper_params = extractor_params['hyper-parameters']
        self.n_lstm_cells = hyper_params['n_lstm_cells']

        if extractor_params['model_flags']['do_dmcnn']:
            self.create_model_dmcnn()
        else:
            self.create_model()

    def create_model(self):
        global keras_argument_model

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        trigger_pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32',
                                   name=u'trigger_position_array')
        argument_pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32',
                                        name=u'argument_position_array')

        anchor_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(trigger_pos_array_input)
        target_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(argument_pos_array_input)

        # Each word is tagged with the SAME event number. Note sure why this event
        # number needs to be repeated, just following the description of the Chen et al
        # DMPooling paper. Embed event in vector space
        if self.use_event_embedding:
            event_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_array')
            event_embedding = Embedding(self.num_event_types - 1, self.position_embedding_vec_len)(event_array_input)

        # trying one shared embedding
        word_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings],
                              embeddings_initializer='glorot_uniform',
                              trainable=self.train_embedding)

        # Input is matrix representing a sentence of self.sent_length tokens, where each token
        # is a vectors of length self.word_vec_length + 6
        # !!!! extra 6 tokens for the two tokens around trigger, and two tokens around role argument
        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window + role_window,), dtype=u'int32', name=u'lex_vector')
        lex_words = word_embedding(lex_input)

        context_input = Input(shape=(self.sent_length,), name=u'word_vector')
        all_words = word_embedding(context_input)

        # Sentence feature input is the result of merging word vectors and embeddings
        if self.use_event_embedding:
            merged = merge([all_words, anchor_embedding, target_embedding, event_embedding], mode=u'concat')
        else:
            merged = merge([all_words, anchor_embedding, target_embedding], mode=u'concat')

        # Note: border_mode='same' to keep output the same width as the input
        conv = Convolution1D(
            self.num_feature_maps,
            self.filter_length,
            border_mode='valid',
            activation='relu'
        )(merged)
        # Dynamically max pool 'conv' result into 3 max value
        maxpool = GlobalMaxPooling1D()(conv)

        # Lexical level feature
        lex_flattened = Flatten()(lex_words)

        # LSTM feature
        input_forward = Input(shape=(self.sent_length,), name=u'word_vector_forward')
        input_backward = Input(shape=(self.sent_length,), name=u'word_vector_backward')
        forward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_forward)
        backward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_backward)

        forward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1),
        )(forward_emb)
        backward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1)
        )(backward_emb)

        # Merge sentence and lexical features
        merged_all = merge([maxpool, lex_flattened, forward_lstm, backward_lstm], mode=u'concat')

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)

        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)

        if self.use_event_embedding:
            keras_argument_model = Model(
                inputs=[
                    context_input,
                    input_forward,
                    input_backward,
                    trigger_pos_array_input,
                    argument_pos_array_input,
                    lex_input,
                    event_array_input
                ],
                output=[
                    out
                ]
            )
        else:
            keras_argument_model = Model(
                inputs=[
                    context_input,
                    input_forward,
                    input_backward,
                    trigger_pos_array_input,
                    argument_pos_array_input,
                    lex_input
                ],
                output=[
                    out
                ]
            )

        keras_argument_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )

    def create_model_dmcnn(self):
        global keras_argument_model

        trigger_pos_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input')
        trigger_e = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                   embeddings_initializer='glorot_uniform')(trigger_pos_input)
        trigger_pos_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_left')
        trigger_e_left = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_left)
        trigger_pos_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_middle')
        trigger_e_middle = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_middle)
        trigger_pos_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_right')
        trigger_e_right = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                     embeddings_initializer='glorot_uniform')(trigger_pos_input_right)

        argument_pos_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input')
        argument_e = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(argument_pos_input)
        argument_pos_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_left')
        argument_e_left = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_left)
        argument_pos_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_middle')
        argument_e_middle = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_middle)
        argument_pos_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'argument_pos_input_right')
        argument_e_right = Embedding(2 * self.sent_length, self.position_embedding_vec_len,
                                       embeddings_initializer='glorot_uniform')(argument_pos_input_right)

        event_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input')
        event_e = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                 embeddings_initializer='glorot_uniform')(event_input)
        event_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_left')
        event_e_left = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_left)
        event_input_middle = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_middle')
        event_e_middle = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_middle)
        event_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'event_input_right')
        event_e_right = Embedding(self.num_event_types, self.position_embedding_vec_len,
                                    embeddings_initializer='glorot_uniform')(event_input_right)

        word_input = Input(shape=(self.sent_length,), name=u'word_input')
        word_e = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                weights=[self.word_embeddings], trainable=self.train_embedding)(word_input)
        word_input_left = Input(shape=(self.sent_length,), name=u'word_input_left')
        word_e_left = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_left)
        word_input_middle = Input(shape=(self.sent_length,), name=u'word_input_middle')
        word_e_middle = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                           weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_middle)
        word_input_right = Input(shape=(self.sent_length,), name=u'word_input_right')
        word_e_right = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                           weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_right)

        merged = merge([word_e, trigger_e, argument_e, event_e], mode=u'concat')
        merged_left = merge([word_e_left, trigger_e_left, argument_e_left, event_e_left], mode=u'concat')
        merged_middle = merge([word_e_middle, trigger_e_middle, argument_e_middle, event_e_middle], mode=u'concat')
        merged_right = merge([word_e_right, trigger_e_right, argument_e_right, event_e_right], mode=u'concat')

        conv_all = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same',
                                 activation='relu')(merged)
        maxpool = GlobalMaxPooling1D()(conv_all)

        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same', activation='relu')

        conv_left = conv(merged_left)
        conv_middle = conv(merged_middle)
        conv_right = conv(merged_right)

        maxpool_left = GlobalMaxPooling1D()(conv_left)

        maxpool_middle = GlobalMaxPooling1D()(conv_middle)

        maxpool_right = GlobalMaxPooling1D()(conv_right)

        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window + role_window,), dtype=u'int32', name=u'lex_input')
        lex_e = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(lex_input)
        lex_flattened = Flatten()(lex_e)

        # LSTM feature
        input_forward = Input(shape=(self.sent_length,), name=u'word_vector_forward')
        input_backward = Input(shape=(self.sent_length,), name=u'word_vector_backward')
        forward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_forward)
        backward_emb = Embedding(
            self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            embeddings_initializer='glorot_uniform',
            trainable=self.train_embedding,
            mask_zero=True
        )(input_backward)

        forward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1)
        )(forward_emb)
        backward_lstm = LSTM(
            self.n_lstm_cells,
            return_sequences=False,
            input_shape=(self.sent_length, 1)
        )(backward_emb)

        merged_all = merge(
            [
                maxpool,
                maxpool_left,
                maxpool_middle,
                maxpool_right,
                lex_flattened,
                forward_lstm,
                backward_lstm
            ],
            mode=u'concat'
        )

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)

        keras_argument_model = Model(
            inputs=[
                word_input,
                word_input_left,
                word_input_middle,
                word_input_right,
                input_forward,
                input_backward,
                trigger_pos_input,
                trigger_pos_input_left,
                trigger_pos_input_middle,
                trigger_pos_input_right,
                argument_pos_input,
                argument_pos_input_left,
                argument_pos_input_middle,
                argument_pos_input_right,
                event_input,
                event_input_left,
                event_input_middle,
                event_input_right,
                lex_input
            ],
            output=[
                out
            ]
        )

        keras_argument_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )


class EmbeddedRoleModel(RoleModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(EmbeddedRoleModel, self).__init__(extractor_params, event_domain, embeddings)
        self.train_embedding = extractor_params['model_flags']['train_embeddings']
        self.use_end_hidden_layer = extractor_params['model_flags'].get('use_end_hidden_layer', False)
        self.end_hidden_layer_depth = extractor_params['hyper-parameters'].get('end_hidden_layer_depth', 0)
        self.end_hidden_layer_nodes = extractor_params['hyper-parameters'].get('end_hidden_layer_nodes', 0)
        self.sent_embeddings = embeddings['word_embeddings'].sent_vec
        self.dep_embeddings = embeddings['dependency_embeddings'].word_vec
        self.num_ne_types = len(event_domain.entity_types)

        self.use_common_entity_name = extractor_params['model_flags'].get('use_common_entity_name', False)
        self.use_dep_emb = extractor_params['model_flags'].get('use_dep_emb', False)
        self.use_position_feat = extractor_params['model_flags'].get('use_position_feat', False)

        self.create_model()

    def create_model(self):
        global keras_argument_model

        trigger_window = 2 * self.neighbor_dist + 1
        role_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window + role_window,), dtype=u'int32', name=u'lex_vector')
        lex_trigger_input = Input(shape=(trigger_window,), dtype=u'int32', name=u'lex_vector_trigger')
        lex_role_input = Input(shape=(role_window,), dtype=u'int32', name=u'lex_vector_role')
        common_word_input = Input(shape=(1,), dtype=u'int32', name=u'common_word')
        arg_dep_input = Input(shape=(1,), dtype=u'int32', name=u'arg_dep_input')
        lex_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings],
                              trainable=False)
        dep_emb = Embedding(
            self.dep_embeddings.shape[0],
            self.dep_embeddings.shape[1],
            weights=[self.dep_embeddings],
            trainable=False
        )


        lex_trigger_emb = Flatten()(lex_emb(lex_trigger_input))
        lex_role_emb = Flatten()(lex_emb(lex_role_input))
        lex_input_emb = Flatten()(lex_emb(lex_input))
        common_word_emb = Flatten()(lex_emb(common_word_input))
        dep_input_emb = Flatten()(dep_emb(arg_dep_input))
        # Lexical level feature
        #lex_flattened = concatenate([lex_trigger_emb, lex_role_emb])
        #lex_flattened = lex_role_emb
        lex_flattened = lex_input_emb

        event_input = Input(shape=(1,), dtype=u'int32', name=u'event_input')
        #35
        event_emb = Dropout(self.dropout)(Embedding(self.num_event_types, self.position_embedding_vec_len,
                              embeddings_initializer='glorot_uniform')(event_input))
        ner_input = Input(shape=(1,), dtype=u'int32', name=u'ne_input')
        #10
        ner_emb = Dropout(self.dropout)(Embedding(self.num_ne_types, self.position_embedding_vec_len,
                              embeddings_initializer='glorot_uniform')(ner_input))

        # these didnt help and are unused
        # sent_input = Input(shape=(1,), dtype=u'int32', name=u'sent_input')
        # sent_emb = Embedding(self.sent_embeddings.shape[0], self.sent_embeddings.shape[1],
        #                      weights=[self.sent_embeddings],
        #                      embeddings_initializer='glorot_uniform',
        #                      trainable=False)(sent_input)
        #
        # pos_input = Input(shape=(2,), dtype=u'int32', name=u'pos_input')
        # pos_emb = Embedding(self.sent_length, self.position_embedding_vec_len,
        #                          embeddings_initializer='glorot_uniform')(pos_input)

        dist_input = Input(shape=(1,), dtype=u'int32', name=u'dist_input')
        dist_emb = Dropout(self.dropout)(Embedding(self.sent_length, self.position_embedding_vec_len,
                            embeddings_initializer='glorot_uniform')(dist_input))

        rel_pos_input = Input(shape=(1,), dtype=u'int32', name=u'pos_input')
        rel_pos_emb = Dropout(self.dropout)(Embedding(4, self.position_embedding_vec_len,
                                 embeddings_initializer='glorot_uniform')(rel_pos_input))

        unique_role_input = Input(shape=(1,), dtype=u'int32', name=u'unique_role_input')
        unique_role_emb = Dropout(self.dropout)(Embedding(2, self.position_embedding_vec_len,
                                 embeddings_initializer='glorot_uniform')(unique_role_input))
        nearest_type_input = Input(shape=(1,), dtype=u'int32', name=u'nearest_type_input')
        nearest_type_emb = Dropout(self.dropout)(Embedding(2, self.position_embedding_vec_len,
                                 embeddings_initializer='glorot_uniform')(nearest_type_input))
        embedding_list = [lex_flattened]

        if self.use_common_entity_name:
            embedding_list.append(common_word_emb)
        if self.use_dep_emb:
            embedding_list.append(dep_input_emb)
        if len(embedding_list) > 1:
            merged_output = concatenate(embedding_list)
        else:
            merged_output = embedding_list[0]
        #
        # merged_output = concatenate(
        #     [
        #         #lex_role_emb,
        #         lex_flattened,
        #         #Flatten()(event_emb),
        #         #Flatten()(pos_emb),
        #         #Flatten()(sent_emb),
        #         #Flatten()(ner_emb),
        #         #Flatten()(rel_pos_emb), # n
        #         #Flatten()(dist_emb),#y
        #         #Flatten()(unique_role_emb),#y
        #         #Flatten()(nearest_type_emb),#n
        #         common_word_emb,#y
        #         dep_input_emb
        #     ]
        # )

        prev_layer = merged_output

        if self.use_end_hidden_layer:
            for i in range(self.end_hidden_layer_depth):
                if i == self.end_hidden_layer_depth - 1:
                    dropout = Dropout(self.dropout)(prev_layer)
                    prev_layer = Dense(
                        self.end_hidden_layer_nodes,
                        activation='relu',
                        kernel_constraint=maxnorm(3)
                    )(dropout)
                else:
                    prev_layer = Dense(
                        self.end_hidden_layer_nodes,
                        activation='relu',
                        kernel_constraint=maxnorm(3)
                    )(prev_layer)

        # 253

        to_output_layer_list = [prev_layer, Flatten()(event_emb)]

        if self.use_position_feat:
            to_output_layer_list.extend(
                [
                    Flatten()(ner_emb),
                    Flatten()(dist_emb),
                    Flatten()(unique_role_emb),
                    Flatten()(rel_pos_emb),
                    Flatten()(nearest_type_emb)
                ]
            )

        prev_layer = concatenate(to_output_layer_list)

        out = Dense(
            self.num_output,
            activation=u'softmax',
            kernel_constraint=maxnorm(3)
        )(prev_layer)


        # model_inputs = [
        #     lex_input,
        #     #pos_input,
        #     #
        #     #lex_trigger_input,
        #     #lex_role_input,
        #     #sent_input,
        #     ner_input,
        #     event_input,
        #     rel_pos_input,
        #     dist_input,
        #     unique_role_input,
        #     nearest_type_input,
        #     common_word_input,
        #     arg_dep_input
        # ]

        model_inputs = [
            lex_input,
            event_input
        ]

        if self.use_position_feat:
            model_inputs.extend(
                [
                    ner_input,
                    rel_pos_input,
                    dist_input,
                    unique_role_input,
                    nearest_type_input,
                ]
            )

        if self.use_common_entity_name:
            model_inputs.append(
                common_word_input,
            )

        if self.use_dep_emb:
            model_inputs.append(
                arg_dep_input
            )

        # model_inputs = [
        #     lex_input,
        #     #pos_input,
        #     #
        #     #lex_trigger_input,
        #     #lex_role_input,
        #     #sent_input,
        #     event_input,
        #     ner_input,
        #     rel_pos_input,
        #     dist_input,
        #     unique_role_input,
        #     nearest_type_input,
        #     common_word_input,
        #     arg_dep_input
        # ]

        keras_argument_model = Model(inputs=model_inputs, output=[out])

        keras_argument_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )


