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

global keras_sentence_model

class SentenceModel(EventExtractionModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(SentenceModel, self).__init__(params, event_domain, embeddings,
                                           batch_size=params.get_int('sentence.batch_size'),
                                           num_feature_maps=params.get_int('sentence.num_feature_maps'))
        self.num_output = len(event_domain.event_types)
        self.positive_weight = params.get_float('sentence.positive_weight')
        self.epoch = params.get_int('sentence.epoch')

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_sentence_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = self.event_domain.get_event_type_index('None')
        sample_weight = np.ones(train_label.shape[0])
        label_argmax = np.argmax(train_label, axis=1)
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight

        #super(TriggerModel, self).fit(train_label, train_data_list, test_label, test_data_list,
        #            sample_weight=sample_weight, max_epoch=self.epoch)
        history = keras_sentence_model.fit(train_data_list, train_label,
                                  sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch,
                                  validation_data=(test_data_list, test_label))
        return history

    def load_keras_model(self, filename=None):
        global keras_sentence_model
        keras_sentence_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_sentence_model
        keras_sentence_model.save(filename)

    def predict(self, test_data_list):
        global keras_sentence_model

        try:
            pred_result = keras_sentence_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'sentence.hdf'))
            print('*** Loaded keras_sentence_model ***')
            pred_result = keras_sentence_model.predict(test_data_list)
        return pred_result

class MaxPoolEmbeddedSentenceModel(SentenceModel):
    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedSentenceModel, self).__init__(params, event_domain, embeddings)
        self.train_embedding = False
        self.create_model()

    def create_model(self):
        global keras_sentence_model

        # Input is vector of embedding indice representing the sentence
        # !!!! + 3
        context_input = Input(shape=(self.sent_length,), dtype=u'int32', name=u'word_vector')
        all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        context_words = MyRange(0, None)(all_words)

        # Sentence feature input is the result of mergeing word vectors and embeddings
        #merged = merge([context_words], mode=u'concat')
        merged = context_words

        # Note: border_mode='same' to keep output the same width as the input
        conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')(merged)

        # Dynamially max pool 'conv' result into 3 max value
        maxpool = GlobalMaxPooling1D()(conv)

        #Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'valid')

        # Input anchor and target words, plus +/- one context words
        # lex_vector = Input(shape=(3,self.word_vec_length), name='lex')

        # Merge sentance and lexcial features
        #merged_all = merge([maxpool], mode=u'concat')
        merged_all = maxpool

        # Dense MLP layer with dropout
        dropout = Dropout(self.dropout)(merged_all)

        out = Dense(self.num_output, activation=u'softmax')(dropout)

        keras_sentence_model = Model(input=[context_input], output=[out])

        keras_sentence_model.compile(optimizer=self.optimizer,
                                loss=u'categorical_crossentropy',
                                metrics=[])
