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

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.model.my_keras import MyRange
from nlplingo.model.my_keras import MySelect
from nlplingo.model.event_cnn import EventExtractionModel

global keras_pair_model

class PairModel(EventExtractionModel):
    def __init__(self, params, event_domain, embeddings, causal_embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type causal_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(PairModel, self).__init__(params, event_domain, embeddings, causal_embeddings,
                                           batch_size=params.get_int('pair.batch_size'),
                                           num_feature_maps=params.get_int('pair.num_feature_maps'))
        self.num_output = 2
        self.positive_weight = params.get_float('pair.positive_weight')
        self.epoch = params.get_int('pair.epoch')
        #self.existing_classes = read_file_to_set(params.get_string('existing_types'))
        #self.new_classes = read_file_to_set(params.get_string('new_types'))           # novel event types

    def fit(self, train_data_list, train_label, test_data_list, test_label):
        global keras_pair_model

        if self.verbosity == 1:
            print('- train_data_list=', train_data_list)
            print('- train_label=', train_label)

        none_label_index = 0
        sample_weight = np.ones(train_label.shape[0])
        #label_argmax = np.argmax(train_label, axis=1)
        label_argmax = train_label
        for i, label_index in enumerate(label_argmax):
            if label_index != none_label_index:
                sample_weight[i] = self.positive_weight

        #super(TriggerModel, self).fit(train_label, train_data_list, test_label, test_data_list,
        #            sample_weight=sample_weight, max_epoch=self.epoch)
        history = keras_pair_model.fit(train_data_list, train_label,
                                  sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=self.epoch)
                                  #validation_data=(test_data_list, test_label))
        return history

    def load_keras_model(self, filename=None):
        global keras_pair_model
        keras_pair_model = keras.models.load_model(filename, self.keras_custom_objects)

    def save_keras_model(self, filename):
        global keras_pair_model
        keras_pair_model.save(filename)

    def predict(self, test_data_list):
        global keras_pair_model

        try:
            pred_result = keras_pair_model.predict(test_data_list)
        except:
            self.load_keras_model(filename=os.path.join(self.model_dir, 'pair.hdf'))
            print('*** Loaded keras_pair_model ***')
            pred_result = keras_pair_model.predict(test_data_list)
        return pred_result

class MaxPoolEmbeddedPairModel(PairModel):
    def __init__(self, params, event_domain, embeddings, causal_embeddings):
        """
        :type params: nlplingo.common.parameters.Parameters
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type causal_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        super(MaxPoolEmbeddedPairModel, self).__init__(params, event_domain, embeddings, causal_embeddings)
        self.train_embedding = False
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        #self.use_trigger = params.get_boolean('use_trigger')
        #self.use_role = params.get_boolean('use_role')
        #self.role_use_head = params.get_boolean('role.use_head')
        self.number_of_hidden_nodes = params.get_int('nn.hidden_nodes')
        self.use_cnn = params.get_boolean('use_cnn')
        self.use_dropout = params.get_boolean('use_dropout')
        self.create_model()

    def create_model(self):
        global keras_pair_model

        pos_input1 = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array1')
        pos_input2 = Input(shape=(self.sent_length,), dtype=u'int32', name=u'position_array2')

        # For each word the pos_array_input defines two coordinates:
        # distance(word,anchor) and distance(word,target).  Embed each distances
        # into an 'embedding_vec_length' dimensional vector space
        # pos_input1 = Input(shape=(2, self.sent_length), dtype=u'int32', name=u'position_array1')
        # anchor_pos_array1 = MySelect(0)(pos_input1)
        # target_pos_array1 = MySelect(1)(pos_input1)
        # anchor_embedding1 = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(anchor_pos_array1)
        # target_embedding1 = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(target_pos_array1)
        #
        # pos_input2 = Input(shape=(2, self.sent_length), dtype=u'int32', name=u'position_array2')
        # anchor_pos_array2 = MySelect(0)(pos_input2)
        # target_pos_array2 = MySelect(1)(pos_input2)
        # anchor_embedding2 = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(anchor_pos_array2)
        # target_embedding2 = Embedding(2 * self.sent_length, self.position_embedding_vec_len)(target_pos_array2)

        trigger_window = 2 * self.neighbor_dist + 1
        # if self.role_use_head:
        #     role_window = 2 * self.neighbor_dist + 1
        # else:
        #     role_window = 2 * self.neighbor_dist
        #
        # if self.use_trigger and trigger_window == 0:
        #     raise ValueError('If using trigger, trigger_window must be >= 0')
        # if self.use_role and role_window == 0:
        #     raise ValueError('If using role, role_window must be >= 0 (you must either use role_head or neighbor_dist > 0)')

        # dep_input1 = Input(shape=(1,), dtype=u'int32', name=u'dep_vector1')
        # dep_input2 = Input(shape=(1,), dtype=u'int32', name=u'dep_vector2')

        # Input is vector of embedding indice representing the sentence
        context_input1 = Input(shape=(self.sent_length + trigger_window,), dtype=u'int32', name=u'word_vector1')
        context_input2 = Input(shape=(self.sent_length + trigger_window,), dtype=u'int32', name=u'word_vector2')

        # context_cinput1 = Input(shape=(self.sent_length + trigger_window + role_window,), dtype=u'int32',
        #                        name=u'word_cvector1')
        # context_cinput2 = Input(shape=(self.sent_length + trigger_window + role_window,), dtype=u'int32',
        #                        name=u'word_cvector2')

        embeddings = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                             weights=[self.word_embeddings], trainable=self.train_embedding)
        # causal_embeddings = Embedding(self.causal_word_embeddings.shape[0], self.causal_word_embeddings.shape[1],
        #                        weights=[self.causal_word_embeddings], trainable=self.train_embedding)

        all_words1 = embeddings(context_input1)
        all_words2 = embeddings(context_input2)

        # all_cwords1 = causal_embeddings(context_cinput1)
        # all_cwords2 = causal_embeddings(context_cinput2)
        #
        # dep_embeddings = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
        #                        weights=[self.word_embeddings], trainable=self.train_embedding)
        #
        # all_dep1 = dep_embeddings(dep_input1)
        # all_dep2 = dep_embeddings(dep_input2)

        #all_words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
        #                      weights=[self.word_embeddings], trainable=self.train_embedding)(context_input)

        context_words1 = MyRange(0, -trigger_window)(all_words1)
        context_words2 = MyRange(0, -trigger_window)(all_words2)

        # merged1 = merge([context_words1, target_embedding1], mode=u'concat')
        # merged2 = merge([context_words2, target_embedding2], mode=u'concat')



        #Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'valid')

        # Input anchor and target words, plus +/- one context words
        # lex_vector = Input(shape=(3,self.word_vec_length), name='lex')

        # Merge sentance and lexcial features
        #merged_all = merge([maxpool1, maxpool2], mode=u'concat')
#        merged_all = keras.layers.multiply([maxpool1, maxpool2])
#        print('type(merged_all)={}'.format(type(merged_all)))

#        bi_dir = Bidirectional(LSTM(100, input_shape=all_words1.shape[1:], return_sequences=True))
#        maxpool1 = bi_dir(all_words1)
#        maxpool2 = bi_dir(all_words2)

        #sim_att = custom_attn.AttentionMMA(merge_mode="concat")([maxpool1, maxpool2])
        #merged_all = BatchNormalization()(sim_att)

        lex_vector1 = MyRange(-trigger_window, None)(all_words1)
        lex_vector2 = MyRange(-trigger_window, None)(all_words2)

        # if self.use_trigger and self.use_role:
        #     lex_vector1 = MyRange(-(trigger_window + role_window), None)(all_words1)
        #     lex_vector2 = MyRange(-(trigger_window + role_window), None)(all_words2)
        #     lex_cvector1 = MyRange(-(trigger_window + role_window), None)(all_cwords1)
        #     lex_cvector2 = MyRange(-(trigger_window + role_window), None)(all_cwords2)
        #     if role_window > 0:
        #         lex_cvector1 = MyRange(0, -role_window)(lex_cvector1)
        #         lex_cvector2 = MyRange(0, -role_window)(lex_cvector2)
        # else:
        #     if self.use_role:
        #         lex_vector1 = MyRange(-role_window, None)(all_words1)
        #         lex_vector2 = MyRange(-role_window, None)(all_words2)
        #         lex_cvector1 = MyRange(-role_window, None)(all_cwords1)
        #         lex_cvector2 = MyRange(-role_window, None)(all_cwords2)
        #     elif self.use_trigger:
        #         lex_vector1 = MyRange(-(trigger_window + role_window), None)(all_words1)
        #         lex_vector2 = MyRange(-(trigger_window + role_window), None)(all_words2)
        #         lex_cvector1 = MyRange(-(trigger_window + role_window), None)(all_cwords1)
        #         lex_cvector2 = MyRange(-(trigger_window + role_window), None)(all_cwords2)
        #         if role_window > 0:
        #             lex_vector1 = MyRange(0, -role_window)(lex_vector1)
        #             lex_vector2 = MyRange(0, -role_window)(lex_vector2)
        #             lex_cvector1 = MyRange(0, -role_window)(lex_cvector1)
        #             lex_cvector2 = MyRange(0, -role_window)(lex_cvector2)

        lex_flattened1 = Flatten()(lex_vector1)
        lex_flattened2 = Flatten()(lex_vector2)

        # lex_cflattened1 = Flatten()(lex_cvector1)
        # lex_cflattened2 = Flatten()(lex_cvector2)
        #
        # dep_flattened1 = Flatten()(all_dep1)
        # dep_flattened2 = Flatten()(all_dep2)



#        merged_all1 = merge([maxpool1, lex_flattened1], mode=u'concat')
#        merged_all2 = merge([maxpool2, lex_flattened2], mode=u'concat')

#        lex_layer = keras.layers.multiply([lex_flattened1, lex_flattened2])
#        lex_out = Dense(100, activation=u'sigmoid')(lex_layer)
#        clex_layer = keras.layers.multiply([lex_cflattened1, lex_cflattened2])
#        clex_out = Dense(100, activation=u'sigmoid')(clex_layer)
#        out = merge([lex_out, clex_out], mode=u'concat')

        #merged_final = keras.layers.multiply([maxpool1, maxpool2])
        #merged_final = keras.layers.multiply([merged_all1, merged_all2])
        #merged_final = merge([lex_flattened1, lex_flattened2], mode=u'concat')

        #merged_final = keras.layers.multiply([lex_flattened1, lex_flattened2])  # use this

        #premerge_1 = merge([lex_flattened1, lex_cflattened1, dep_flattened1], mode=u'concat')
        #premerge_2 = merge([lex_flattened2, lex_cflattened2, dep_flattened2], mode=u'concat')
        # premerge_1 = merge([lex_flattened1, dep_flattened1], mode=u'concat')
        # premerge_2 = merge([lex_flattened2, dep_flattened2], mode=u'concat')

        # Note: border_mode='same' to keep output the same width as the input
        if self.use_cnn:
            conv = Convolution1D(self.num_feature_maps, self.filter_length, border_mode=u'same')
            conv1 = conv(context_words1)
            conv2 = conv(context_words2)
            maxpool1 = GlobalMaxPooling1D()(conv1)
            maxpool2 = GlobalMaxPooling1D()(conv2)
            merged1 = merge([maxpool1, lex_flattened1], mode=u'concat')
            merged2 = merge([maxpool2, lex_flattened2], mode=u'concat')
            merged_final = keras.layers.multiply([merged1, merged2])
        else:
            merged_final = keras.layers.multiply([lex_flattened1, lex_flattened2])  # use this
        #merged_final = keras.layers.multiply([premerge_1, premerge_2])  # use this
        #merged_final = merge([premerge_1, premerge_2], mode=u'concat')

        # Dense MLP layer with dropout
        #dropout = Dropout(self.dropout)(merged_final)

        if self.use_dropout:
            dcn_input = Dropout(self.dropout)(merged_final)
        else:
            dcn_input = merged_final

        if self.number_of_hidden_nodes == 0:
            out_final = Dense(1, activation=u'sigmoid')(dcn_input)
        else:
            out = Dense(self.number_of_hidden_nodes, activation=u'sigmoid')(dcn_input)
            #dropout2 = Dropout(self.dropout)(out)
            out_final = Dense(1, activation=u'sigmoid')(out)

        #keras_pair_model = Model(input=[context_input1, context_input2, pos_input1, pos_input2, context_cinput1, context_cinput2, dep_input1, dep_input2], output=[out_final])
        keras_pair_model = Model(input=[context_input1, context_input2, pos_input1, pos_input2], output=[out_final])

        keras_pair_model.compile(optimizer='adadelta',
                                loss=u'binary_crossentropy',
                                metrics=['accuracy'])

