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

global keras_wordpair_model


class WordPairModel(EventExtractionModel):
    def __init__(self, params, embeddings, event_domain):
        """
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(WordPairModel, self).__init__(params, event_domain, embeddings,
                                           batch_size=params.get_int('wordpair.batch_size'),
                                           num_feature_maps=params.get_int('wordpair.num_feature_maps'))
        #self.num_output = 2
        self.positive_weight = params.get_float('wordpair.positive_weight')
        self.epoch = params.get_int('wordpair.epoch')

    def fit(self, train_data_list, train_label):
        global keras_wordpair_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = 0
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = train_label
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight

        history = keras_wordpair_model.fit(train_data_list, train_label,
                                  sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch)
                                  #validation_data=(test_data_list, test_label))
        return history

    def load_keras_model(self, filename=None):
        global keras_wordpair_model
        keras_wordpair_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_wordpair_model
        keras_wordpair_model.save(filename)
        print(keras_wordpair_model.summary())

    def predict(self, test_data_list):
        global keras_wordpair_model

        try:
            pred_result = keras_wordpair_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'wordpair.hdf'))
            print('*** Loaded keras_wordpair_model ***')
            pred_result = keras_wordpair_model.predict(test_data_list)
        return pred_result

class MaxPoolEmbeddedWordPairModel(WordPairModel):
    def __init__(self, params, embeddings, event_domain):
        """
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedWordPairModel, self).__init__(params, embeddings, event_domain)
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        self.number_of_hidden_nodes = [int(node) for node in params.get_list('nn.hidden_nodes')]
        """:type: list[int]"""
        self.use_cnn = params.get_boolean('use_cnn')
        self.use_dropout = params.get_boolean('use_dropout')
        self.train_embedding = False
        self.create_model()

    def create_model(self):
        global keras_wordpair_model

        wordpair_input = Input(shape=(2,), dtype=u'int32', name=u'wordpair_vector')
        wordpair_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.train_embedding)(wordpair_input)
        wordpair_flattened = Flatten()(wordpair_embedding)

        # Input is vector of embedding indices representing the sentence
        window_size = 2 * self.neighbor_dist + 1
        #context_input = Input(shape=(self.sent_length + window_size,), dtype=u'int32', name=u'word_vector')
        #all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
        #                     weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        lexical_input = Input(shape=(window_size,), dtype=u'int32', name=u'lexical_vector')
        lex_vector = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=self.train_embedding)(lexical_input)


        # surrounding words in local window
        #lex_vector = MyRange(-window_size, None)(all_words)
        lex_flattened = Flatten()(lex_vector)

        if self.use_dropout:
            wp_input = Dropout(self.dropout)(wordpair_flattened)
            lex_input = Dropout(self.dropout)(lex_flattened)
        else:
            wp_input = wordpair_flattened
            lex_input = lex_flattened

        wp_hidden = Dense(self.number_of_hidden_nodes[0], activation=u'sigmoid')(wp_input)
        lex_hidden = Dense(self.number_of_hidden_nodes[0], activation=u'sigmoid')(lex_input)

        out = keras.layers.multiply([wp_hidden, lex_hidden])
        out_final = Dense(1, activation=u'sigmoid')(out)

        keras_wordpair_model = Model(input=[lexical_input, wordpair_input], output=[out_final])

        keras_wordpair_model.compile(optimizer='adadelta', loss=u'binary_crossentropy', metrics=['accuracy'])

	# we now use the above multiply model

    # def create_model(self):
    #     global keras_wordpair_model
    #
    #     wordpair_input = Input(shape=(2,), dtype=u'int32', name=u'wordpair_vector')
    #     wordpair_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
    #                           weights=[self.word_embeddings], trainable=self.train_embedding)(wordpair_input)
    #     wordpair_flattened = Flatten()(wordpair_embedding)
    #
    #     # Input is vector of embedding indices representing the sentence
    #     window_size = 2 * self.neighbor_dist + 1
    #     context_input = Input(shape=(self.sent_length + window_size,), dtype=u'int32', name=u'word_vector')
    #     all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
    #                          weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)
    #
    #     # surrounding words in local window
    #     lex_vector = MyRange(-window_size, None)(all_words)
    #     lex_flattened = Flatten()(lex_vector)
    #
    #     if self.use_cnn:
    #         # For each word the pos_array_input defines the distance to the target work.
    #         # Embed each distance into an 'embedding_vec_length' dimensional vector space
    #         pos_array_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array')
    #         # the input dimension is 2*self.sent_length, because the range of numbers go from min=0 to max=2*self.sent_length
    #         pos_embedding = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(pos_array_input)
    #
    #         context_words = MyRange(0, -window_size)(all_words)
    #
    #         merged = merge([context_words, pos_embedding], mode=u'concat')
    #         conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')(merged)
    #         maxpool = GlobalMaxPooling1D()(conv)
    #
    #         merged_final = merge([wordpair_flattened, lex_flattened, maxpool], mode=u'concat')
    #     else:
    #         merged_final = merge([wordpair_flattened, lex_flattened], mode=u'concat')
    #
    #     if self.use_dropout:
    #         dcn_input = Dropout(self.dropout)(merged_final)
    #     else:
    #         dcn_input = merged_final
    #
    #     if self.number_of_hidden_nodes[0] == 0:
    #         out_final = Dense(1, activation=u'sigmoid')(dcn_input)
    #     elif self.number_of_hidden_nodes[1] == 0:
    #         out = Dense(self.number_of_hidden_nodes[0], activation=u'sigmoid')(dcn_input)
    #         out_final = Dense(1, activation=u'sigmoid')(out)
    #     else:
    #         out1 = Dense(self.number_of_hidden_nodes[0], activation=u'sigmoid')(dcn_input)
    #         out = Dense(self.number_of_hidden_nodes[1], activation=u'sigmoid')(out1)
    #         out_final = Dense(1, activation=u'sigmoid')(out)
    #
    #     if self.use_cnn:
    #         keras_wordpair_model = Model(input=[context_input, wordpair_input, pos_array_input], output=[out_final])
    #     else:
    #         keras_wordpair_model = Model(input=[context_input, wordpair_input], output=[out_final])
    #
    #     keras_wordpair_model.compile(optimizer='adadelta', loss=u'binary_crossentropy', metrics=['accuracy'])
