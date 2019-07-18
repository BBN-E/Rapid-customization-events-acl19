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
from keras.layers import concatenate
from keras.layers import LSTM


from nlplingo.model.my_keras import MyRange
from nlplingo.model.my_keras import MySelect
from nlplingo.model.event_cnn import EventExtractionModel

global keras_trigger_model
from keras.constraints import maxnorm

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

class TriggerModel(EventExtractionModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(TriggerModel, self).__init__(params, event_domain, embeddings)

        hyper_params = params['hyper-parameters']
        self.positive_weight = hyper_params['positive_weight']
        self.epoch = hyper_params['epoch']
        self.early_stopping = hyper_params.get('early_stopping', False)
        self.number_of_feature_maps = hyper_params.get('number_of_feature_maps', 0)  # number of convolution feature maps
        self.batch_size = hyper_params['batch_size']

        self.position_embedding_vector_length = hyper_params['position_embedding_vector_length']
        self.filter_lengths = hyper_params.get('cnn_filter_lengths', 0)
        """:type: list[int]"""
        self.dropout = hyper_params['dropout']

        self.entity_embedding_vector_length = hyper_params['entity_embedding_vector_length']
        self.use_bio_index = params['model_flags'].get('use_bio_index', False)
        self.use_lex_info = params['model_flags'].get('use_lex_info', False)

        self.number_of_entity_bio_types = len(event_domain.entity_bio_types)

        self.num_output = len(event_domain.event_types)
        self.model = None

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_trigger_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = self.event_domain.get_event_type_index('None')
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight

        callbacks = None
        if self.early_stopping:
            callbacks = [early_stopping]

        if len(test_label) > 0:
            history = keras_trigger_model.fit(
                train_data_list,
                train_label,
                sample_weight=sample_weight,
                batch_size=self.batch_size,
                nb_epoch=self.epoch,
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
                batch_size=self.batch_size,
                nb_epoch=self.epoch,
                callbacks=callbacks
            )
        return history

    # def fit(self, train_data_list, train_label, test_data_list, test_label):
    #     if self.verbosity == 1:
    #         print('- train_data_list=', train_data_list)
    #         print('- train_label=', train_label)
    #
    #     none_label_index = self.event_domain.get_event_type_index('None')
    #     sample_weight = np.ones(train_label.shape[0])
    #     label_argmax = np.argmax(train_label, axis=1)
    #     for i, label_index in enumerate(label_argmax):
    #         if label_index != none_label_index:
    #             sample_weight[i] = self.positive_weight
    #
    #     if len(test_label) > 0:
    #         history = self.model.fit(train_data_list, train_label,
    #                               sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch,
    #                               validation_data=(test_data_list, test_label), callbacks=[early_stopping])
    #     else:
    #         history = self.model.fit(train_data_list, train_label,
    #                                           sample_weight=sample_weight,
    #                                           batch_size=self.batch_size, nb_epoch=self.epoch, callbacks=[early_stopping])
    #     return history


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
    def __init__(self, params, event_domain, embeddings):
        """
        :type params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(CNNTriggerModel, self).__init__(params, event_domain, embeddings)
        self.neighbor_dist = params['hyper-parameters']['neighbor_distance']
        self.train_embedding = params['model_flags']['train_embeddings']
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        # For each word the pos_array_input defines the distance to the target work.
        # Embed each distance into an 'embedding_vec_length' dimensional vector space
        pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array')
        # the input dimension is 2*self.sent_length, because the range of numbers go from min=0 to max=2*self.sent_length
        pos_embedding = Embedding(2*self.sent_length, self.position_embedding_vector_length)(pos_array_input)

        # Input is vector of embedding indice representing the sentence
        # !!!! + 3
        window_size = 2 * self.neighbor_dist + 1

        sentence_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'sentence_vector')

        text_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=self.train_embedding, name=u'sentence_embedding',dropout=0.3)

        sentence_words = text_emb(sentence_input)
        #context_words = MyRange(0, -window_size)(sentence_words)

        outputs_to_merge_1 = [sentence_words, pos_embedding]

        # Sentence feature input is the result of mergeing word vectors and embeddings
        entity_array_input = None
        if self.use_bio_index:
            entity_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'entity_array')
            entity_embedding = Embedding(self.number_of_entity_bio_types, self.entity_embedding_vector_length)(entity_array_input)
            #merged = merge([sentence_words, pos_embedding, entity_embedding], mode=u'concat')
            outputs_to_merge_1.append(entity_embedding)

        merged = concatenate(outputs_to_merge_1, axis=-1)

        # Note: border_mode='same' to keep output the same width as the input
        maxpools = []
        for filter_length in self.filter_lengths:
            conv = Convolution1D(self.number_of_feature_maps, filter_length, border_mode=u'same', activation='relu')(merged)
            maxpools.append(GlobalMaxPooling1D()(conv))

        #Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'valid')

        # Input anchor and target words, plus +/- one context words
        # lex_vector = Input(shape=(3,self.word_vec_length), name='lex')

        outputs_to_merge_2 = []
        outputs_to_merge_2.extend(maxpools)

        lex_input = None
        if self.use_lex_info:

            lex_input = Input(shape=(window_size,), dtype=u'int32', name=u'lex_vector')
            #lex_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            #                          weights=[self.word_embeddings], trainable=self.train_embedding, name=u'lex_embedding')(lex_input)

            lex_words = text_emb(lex_input)

            # Input indicating the position of the target
            # pos_index_input = Input(shape=(1,), dtype='int32', name='position_index')
            #lex_vector = MyRange(-window_size, None)(all_words)

            # Lexical level feature
            lex_flattened = Flatten()(lex_words)

            # Merge sentance and lexcial features
            #merged_all = merge(maxpools + [lex_flattened], mode=u'concat')
            outputs_to_merge_2.append(lex_flattened)

        merged_all = concatenate(outputs_to_merge_2)

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)

        out = Dense(self.num_output, activation=u'softmax')(dropout)

        model_inputs = [
            sentence_input
        ]

        if self.use_lex_info:
            model_inputs.append(lex_input)

        model_inputs.append(pos_array_input)

        if self.use_bio_index:
            model_inputs.append(entity_array_input)
        keras_trigger_model = Model(inputs=model_inputs, output=[out])
        keras_trigger_model.compile(optimizer=self.optimizer,
                                loss=u'categorical_crossentropy',
                                metrics=[])

        self.model = keras_trigger_model

class EmbeddedTriggerModel(TriggerModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(EmbeddedTriggerModel, self).__init__(extractor_params, event_domain, embeddings)
        self.neighbor_dist = extractor_params['hyper-parameters']['neighbor_distance']
        self.train_embedding = extractor_params['model_flags']['train_embeddings']
        self.use_end_hidden_layer = extractor_params['model_flags'].get('use_end_hidden_layer', False)
        self.end_hidden_layer_depth = extractor_params['hyper-parameters'].get('end_hidden_layer_depth', 0)
        self.end_hidden_layer_nodes = extractor_params['hyper-parameters'].get('end_hidden_layer_nodes', 0)
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        # Input is vector of embedding indice representing the sentence
        # !!!! + 3
        window_size = 2 * self.neighbor_dist + 1

        text_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=False, name=u'sentence_embedding')

        lex_input = Input(shape=(window_size,), dtype=u'int32', name=u'lex_vector')
            #lex_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            #                          weights=[self.word_embeddings], trainable=self.train_embedding, name=u'lex_embedding')(lex_input)

        lex_words = text_emb(lex_input)

            # Input indicating the position of the target
            # pos_index_input = Input(shape=(1,), dtype='int32', name='position_index')
            #lex_vector = MyRange(-window_size, None)(all_words)

            # Lexical level feature
        lex_flattened = Flatten()(lex_words)

        # 128

        prev_layer = lex_flattened

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

        out = Dense(self.num_output, activation=u'softmax')(prev_layer)

        model_inputs = []
        model_inputs.append(lex_input)

        keras_trigger_model = Model(inputs=model_inputs, output=[out])
        keras_trigger_model.compile(optimizer=self.optimizer,
                                loss=u'categorical_crossentropy',
                                metrics=[])

        self.model = keras_trigger_model

class OnlineEmbeddedTriggerModel(TriggerModel):
    def __init__(self, extractor_params, event_domain, embeddings):
        """
        :type params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(OnlineEmbeddedTriggerModel, self).__init__(extractor_params, event_domain, embeddings)
        self.neighbor_dist = extractor_params['hyper-parameters']['neighbor_distance']
        self.train_embedding = extractor_params['model_flags']['train_embeddings']
        self.use_end_hidden_layer = extractor_params['model_flags'].get('use_end_hidden_layer', False)
        self.end_hidden_layer_depth = extractor_params['hyper-parameters'].get('end_hidden_layer_depth', 0)
        self.end_hidden_layer_nodes = extractor_params['hyper-parameters'].get('end_hidden_layer_nodes', 0)
        self.embeddings = embeddings
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        # Input is vector of embedding indice representing the sentence
        # !!!! + 3
        window_size = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(self.embeddings['word_embeddings'].vector_length * window_size,), dtype=u'float32', name=u'lex_vector')
            #lex_emb = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
            #                          weights=[self.word_embeddings], trainable=self.train_embedding, name=u'lex_embedding')(lex_input)



            # Input indicating the position of the target
            # pos_index_input = Input(shape=(1,), dtype='int32', name='position_index')
            #lex_vector = MyRange(-window_size, None)(all_words)

            # Lexical level feature
        lex_flattened = lex_input

        # 128

        prev_layer = lex_flattened

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

        out = Dense(self.num_output, activation=u'softmax')(prev_layer)

        model_inputs = []
        model_inputs.append(lex_input)

        keras_trigger_model = Model(inputs=model_inputs, output=[out])
        keras_trigger_model.compile(optimizer=self.optimizer,
                                loss=u'categorical_crossentropy',
                                metrics=[])

        self.model = keras_trigger_model


class PiecewiseCNNTriggerModel(TriggerModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(PiecewiseCNNTriggerModel, self).__init__(params, event_domain, embeddings)
        self.neighbor_dist = params['hyper-parameters']['neighbor_distance']
        self.train_embedding = params['model_flags']['train_embeddings']
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        position_embedding = Embedding(2 * self.sent_length, self.position_embedding_vector_length,
                              embeddings_initializer='glorot_uniform')
        trigger_pos_input_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_left')
        trigger_e_left = position_embedding(trigger_pos_input_left)
        trigger_pos_input_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'trigger_pos_input_right')
        trigger_e_right = position_embedding(trigger_pos_input_right)

        word_input_left = Input(shape=(self.sent_length,), name=u'word_input_left')
        word_e_left = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_left)
        word_input_right = Input(shape=(self.sent_length,), name=u'word_input_right')
        word_e_right = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                 weights=[self.word_embeddings], trainable=self.train_embedding)(word_input_right)

        if self.use_bio_index:
            entity_array_left = Input(shape=(self.sent_length,), dtype=u'int32', name=u'entity_array_left')
            entity_embedding_left = Embedding(2*self.sent_length, self.entity_embedding_vector_length)(entity_array_left)
            entity_array_right = Input(shape=(self.sent_length,), dtype=u'int32', name=u'entity_array_right')
            entity_embedding_right = Embedding(2 * self.sent_length, self.entity_embedding_vector_length)(entity_array_right)
            merged_left = merge([word_e_left, trigger_e_left, entity_embedding_left], mode=u'concat')
            merged_right = merge([word_e_right, trigger_e_right, entity_embedding_right], mode=u'concat')
        else:
            merged_left = merge([word_e_left, trigger_e_left], mode=u'concat')
            merged_right = merge([word_e_right, trigger_e_right], mode=u'concat')

        maxpools_left = []
        maxpools_right = []
        for filter_length in self.filter_lengths:
            conv = Convolution1D(self.number_of_feature_maps, filter_length, border_mode=u'same', activation='relu')
            conv_left = conv(merged_left)
            conv_right = conv(merged_right)
            maxpools_left.append(GlobalMaxPooling1D()(conv_left))
            maxpools_right.append(GlobalMaxPooling1D()(conv_right))

        trigger_window = 2 * self.neighbor_dist + 1

        lex_input = Input(shape=(trigger_window,), dtype=u'int32', name=u'lex_vector')
        lex_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                   weights=[self.word_embeddings], trainable=self.train_embedding)(lex_input)

        # Lexical level feature
        lex_flattened = Flatten()(lex_words)

        merged_all = merge(maxpools_left + maxpools_right + [lex_flattened],
                           mode=u'concat')

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)
        out = Dense(self.num_output, activation='softmax', kernel_constraint=maxnorm(3))(dropout)

        if self.use_bio_index:
            keras_trigger_model = Model(
                input=[word_input_left, word_input_right, trigger_pos_input_left, trigger_pos_input_right, lex_input,
                       entity_array_left, entity_array_right], output=[out])
        else:
            keras_trigger_model = Model(
                input=[word_input_left, word_input_right, trigger_pos_input_left, trigger_pos_input_right, lex_input],
                output=[out])

        keras_trigger_model.compile(optimizer=self.optimizer, loss=u'categorical_crossentropy', metrics=[])

        self.model = keras_trigger_model


class LSTMTriggerModel(TriggerModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(LSTMTriggerModel, self).__init__(params, event_domain, embeddings)
        self.train_embedding = params['model_flags']['train_embeddings']
        self.n_lstm_cells = params['hyper-parameters']['n_lstm_cells']
        self.create_model()

    def create_model(self):
        global keras_trigger_model

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

        keras_trigger_model = Model(
            input=[input_forward, input_backward],
            output=[out]
        )

        keras_trigger_model.compile(
            optimizer=self.optimizer,
            loss=u'categorical_crossentropy',
            metrics=[]
        )

        self.model = keras_trigger_model


class BiLSTMCNNTriggerModel(TriggerModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(BiLSTMCNNTriggerModel, self).__init__(params, event_domain, embeddings)
        self.neighbor_dist = params['hyper-parameters']['neighbor_distance']
        self.train_embedding = params['model_flags']['train_embeddings']
        self.n_lstm_cells = params['hyper-parameters']['n_lstm_cells']
        self.create_model()

    def create_model(self):
        global keras_trigger_model

        # For each word the pos_array_input defines the distance to the target work.
        # Embed each distance into an 'embedding_vec_length' dimensional vector space
        pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array')
        # the input dimension is 2*self.sent_length, because the range of numbers go from min=0 to max=2*self.sent_length
        pos_embedding = Embedding(2*self.sent_length, self.position_embedding_vector_length)(pos_array_input)

        # Input is vector of embedding indice representing the sentence
        # !!!! + 3
        window_size = 2 * self.neighbor_dist + 1

        sentence_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'sentence_vector')
        sentence_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=self.train_embedding, name=u'sentence_embedding')(sentence_input)

        #context_words = MyRange(0, -window_size)(sentence_words)

        # Sentence feature input is the result of mergeing word vectors and embeddings
        if self.use_bio_index:
            entity_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'entity_array')
            entity_embedding = Embedding(self.number_of_entity_bio_types, self.entity_embedding_vector_length)(entity_array_input)
            #merged = merge([sentence_words, pos_embedding, entity_embedding], mode=u'concat')
            merged = concatenate([sentence_words, pos_embedding, entity_embedding], axis=-1)
        else:
            #merged = merge([sentence_words, pos_embedding], mode=u'concat')
            merged = concatenate([sentence_words, pos_embedding], axis=-1)

        # Note: border_mode='same' to keep output the same width as the input
        maxpools = []
        for filter_length in self.filter_lengths:
            conv = Convolution1D(self.number_of_feature_maps, filter_length, border_mode=u'same', activation='relu')(merged)
            maxpools.append(GlobalMaxPooling1D()(conv))

        #Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'valid')

        # Input anchor and target words, plus +/- one context words

        lex_input = Input(shape=(window_size,), dtype=u'int32', name=u'lex_vector')
        lex_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                   weights=[self.word_embeddings], trainable=self.train_embedding, name=u'lex_embedding')(lex_input)

        # lstm
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
        # end lstm

        # Lexical level feature
        lex_flattened = Flatten()(lex_words)

        # Merge sentence and lexical features

        merged_all = concatenate(maxpools + [lex_flattened, forward_lstm, backward_lstm], axis=-1)


        # Dense MLP layer with dropout
        dropout_1 = Dropout(self.dropout)(merged_all)
        fc_1 = Dense(128, activation='sigmoid')(dropout_1)
        dropout_2 = Dropout(self.dropout)(fc_1)
        fc_2 = Dense(128, activation='sigmoid')(dropout_2)

        out = Dense(self.num_output, activation=u'softmax')(fc_2)

        if self.use_bio_index:
            keras_trigger_model = Model(input=[sentence_input, input_forward, input_backward, lex_input, pos_array_input, entity_array_input], output=[out])
        else:
            keras_trigger_model = Model(input=[sentence_input, input_forward, input_backward, lex_input, pos_array_input], output=[out])

        keras_trigger_model.compile(optimizer=self.optimizer,
                                loss=u'categorical_crossentropy',
                                metrics=[])

        self.model = keras_trigger_model
