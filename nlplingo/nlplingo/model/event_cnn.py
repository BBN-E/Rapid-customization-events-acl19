from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import json
from keras.optimizers import Adadelta, SGD, RMSprop

from nlplingo.model.my_keras import MyRange
from nlplingo.model.my_keras import MySelect

global keras_trigger_model
global keras_argument_model
global keras_sentence_model
global keras_pair_model


class EventExtractionModel(object):
    verbosity = 0

    keras_custom_objects = {
        u'MyRange': MyRange,
        u'MySelect': MySelect,
        #u'ShiftPosition': ShiftPosition,
        #u'DynamicMultiPooling': DynamicMultiPooling,
        #u'DynamicMultiPooling3': DynamicMultiPooling3,
    }

    def __init__(self, params, event_domain, embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type model_name: str
        """

        self.event_domain = event_domain
        self.num_event_types = len(event_domain.event_types)
        self.num_role_types = len(event_domain.event_roles)
        self.num_ne_types = len(event_domain.entity_types)

        # Input consists of two parts sentence feature and lexical feature.
        # Lexical feature is the trigger token and its surrounding context tokens.

        # Sentence feature is all the tokens in the sentences and the distance
        #  of each token to the trigger token.

        self.sent_length = params['max_sentence_length']   # Number of tokens per sentence
        self.word_vec_length = 1                    # because we use word vector index



        self.word_embeddings = embeddings['word_embeddings'].word_vec
        """:type: numpy.ndarray"""
        # self.num_lexical_tokens = 3                 # number of lexical tokens

        ##if causal_embeddings is not None:
        #    self.causal_word_embeddings = causal_embeddings.word_vec
        #    """:type: numpy.ndarray"""
        #else:
        #    self.causal_word_embeddings = None

        self.optimizer = self._configure_optimizer(params)

        self.data_keys = []
        #self.keras_model = None
        #self.keras_model_filename = None
        self.num_output = None

        self.model_dir = None

    def _configure_optimizer(self, params):

        optimizer_params = params.get('optimizer', dict())
        tunable_params = {}
        if optimizer_params.get('name') == 'SGD':
            tunable_params = {
                'name': 'SGD',
                'lr': optimizer_params.get('lr', 0.01),
                'momentum': optimizer_params.get('momentum', 0.0),
                'decay': optimizer_params.get('decay', 0.0),
                'nesterov': optimizer_params.get('nesterov', False)
            }
            optimizer = SGD(
                lr=tunable_params['lr'],
                momentum=tunable_params['momentum'],
                decay=tunable_params['decay'],
                nesterov=tunable_params['nesterov']
            )
        elif optimizer_params.get('name') == 'RMSprop':
            tunable_params = {
                'name': 'RMSprop',
                'lr': optimizer_params.get('lr', 0.001),
                'rho': optimizer_params.get('rho', 0.9),
                'epsilon': optimizer_params.get('epsilon', None),
                'decay': optimizer_params.get('decay', 0.0)
            }
            optimizer = RMSprop(
                lr=tunable_params['lr'],
                rho=tunable_params['rho'],
                epsilon=tunable_params['epsilon'],
                decay=tunable_params['decay']
            )
        elif optimizer_params.get('name') == 'Adam':
            tunable_params = {
                'name': 'Adam',
                'lr': optimizer_params.get('lr', 0.001)
            }
            optimizer = Adam(
                lr=tunable_params['lr']
            )
        else:
            tunable_params = {
                'name': 'Adadelta',
                'lr': optimizer_params.get('lr', 0.1),
                'rho': optimizer_params.get('rho', 0.95),
                'epsilon': optimizer_params.get('epsilon', 1e-6),
                'decay': optimizer_params.get('decay', 0.0)
            }
            # Default Adadelta
            optimizer = Adadelta(
                lr=tunable_params['lr'],
                rho=tunable_params['rho'],
                epsilon=tunable_params['epsilon']
            )
        print('=== Optimization parameters ===')
        print(json.dumps(tunable_params, sort_keys=True, indent=4))
        print('=== Optimization parameters ===')
        return optimizer

    #def get_metric_list(self):
    #    return []

    def create_model(self):
        pass

    def __getstate__(self):
        u"""Defines what is to be pickled.
        Keras models cannot be pickled. Should call save_keras_model() and load_keras_model() separately.
        The sequence is :
        obj.save_keras_model('kerasFilename')
        pickle.dump(obj, fileHandle)
        ...
        obj = pickle.load(fileHandle)
        obj.load_keras_model()"""

        # Create state without self.keras_model
        state = dict(self.__dict__)
        #state.pop(u'keras_model')   # probably not needed anymore, now that we've made keras_model global
        return state

    def __setstate__(self, state):
        global keras_trigger_model
        global keras_argument_model
        # Reload state for unpickling
        self.__dict__ = state
        keras_trigger_model = None
        keras_argument_model = None

    # def save_keras_model(self, filename):
    #     self.keras_model_filename = filename
    #     self.keras_model.save(filename)

    # def load_keras_model(self, filename=None):
    #     global keras_model
    #     keras_model = keras.models.load_model(filename, self.keras_custom_objects)

    # def create_word_embedding_layer(self, trainable=False):
    #     return Embedding(self.word_embedding_array.shape[0], self.word_embedding_array.shape[1],
    #                      weights=[self.word_embedding_array], trainable=trainable)

    # def fit(self, train_label, train_data_list, test_label, test_data_list, sample_weight=None, max_epoch=10):
    #     global keras_model
    #
    #     if self.verbosity == 1:
    #         print('\nevent_cnn.py : EventExtractionModel.fit()')
    #         print('- train_label=', train_label)
    #         print('- train_data_list=', train_data_list)
    #         print('- test_label=', test_label)
    #         print('- test_data_list=', test_data_list)
    #         print('- sample_weight=', sample_weight)
    #         print('- epoch=', max_epoch)
    #
    #     if sample_weight is None:
    #         sample_weight = np.ones((train_label.shape[0]))
    #
    #     history = keras_model.fit(train_data_list, train_label,
    #                                    sample_weight=sample_weight, batch_size=self.batch_size, nb_epoch=max_epoch,
    #                                    validation_data=(test_data_list, test_label))
    #     return history

    # def predict(self, test_data_list):
    #     global keras_model
    #
    #     try:
    #         pred_result = keras_model.predict(test_data_list)
    #     except:
    #         self.load_keras_model(filename=os.path.join(self.model_dir, 'trigger.hdf'))
    #         print('*** Loaded keras_model ***')
    #         pred_result = keras_model.predict(test_data_list)
    #
    #     return pred_result

    # @classmethod
    # def run_model(cls, train, test, weights=range(1, 11), epoch=10, num_skipped_events=0, model_params={}):
    #     all_results = []
    #     best_f1 = 0
    #     best_model = None
    #     for weight in weights:
    #         print(u'Weight={0}'.format(weight))
    #         model = cls(**model_params)
    #         history = model.fit(train, test, event_weight=weight, epoch=epoch)
    #         # print('training')
    #         print(u'testing')
    #         pred = model.predict(test)
    #         evala = ace_eval(pred, test[u'label'], model.num_output, num_skipped_events=num_skipped_events)
    #         if evala[u'f1'] > best_f1:
    #             best_f1 = evala[u'f1']
    #             best_model = model
    #         result = {}
    #         result[u'history'] = history
    #         result[u'pred'] = pred
    #         result[u'eval'] = evala
    #         result[u'description'] = u'weight={0}'.format(weight)
    #         all_results.append(result)
    #     return (all_results, best_model)

    # def fit_weight_boost(self, train_label, train_data_list, test_label, test_data_list,
    #         event_weight=1, sample_weight=None, max_epoch=5):
    #     print('\ncnn.py : EventExtractionModel.fit()')
    #     print('- train_label=', train_label)
    #     print('- train_data_list=', train_data_list)
    #     print('- test_label=', test_label)
    #     print('- test_data_list=', test_data_list)
    #     print('- event_weight=', event_weight)
    #     print('- sample_weight=', sample_weight)
    #     print('- epoch=', max_epoch)
    #
    #     if sample_weight is None:
    #         sample_weight = np.ones((train_label.shape[0]))
    #         sample_weight[train_label[:, self.num_output - 1] != 1] = event_weight
    #
    #     print('* invoking keras_model.fit_weight_boost()')
    #     print('- self.keras_model=', self.keras_model)
    #
    #     weight = np.ones(train_label.shape[0])
    #     for epoch in range(1, max_epoch + 1):
    #         history = self.keras_model.fit(train_data_list, train_label,
    #                                    sample_weight=weight, batch_size=self.batch_size, nb_epoch=5,
    #                                    validation_data=(test_data_list, test_label))
    #         predictions = self.predict(test_data_list)
    #         evaluate_f1(predictions, test_label, self.event_domain.get_event_type_index('None'))
    #
    #         if epoch < max_epoch:
    #             predictions = self.predict(train_data_list)
    #             pred_train_argmax = np.argmax(predictions, axis=1)
    #             for i in range(len(weight)):
    #                 if train_label[i, pred_train_argmax[i]] == 1:
    #                     weight[i] /= 1.2  # I got this correct, so I'm decreasing the weight
    #                 else:
    #                     weight[i] *= 1.2  # I got this wrong, so I'm increasing the weight
    #     return history

    # @classmethod
    # def run_weight_boost_model(cls, train, test, epoch_max=10, weight_factor=1.2,
    #                            num_skipped_events=None, num_total_events=None, outfile_pattern=None, model_params={}):
    #     print('\ntrigger_class_dm_pool_model.py : EventExtractionModel().run_weight_boost_model()')
    #     print('- train=', train)
    #     print('- test=', test)
    #     print('- epoch_max=', epoch_max)
    #     print('- weight_factor=', weight_factor)
    #     print('- num_skippepd_events=', num_skipped_events)
    #     print('- num_total_events=', num_total_events)
    #     print('- outfile_pattern=', outfile_pattern)
    #     print('- model_params=', model_params)
    #
    #     model = cls(**model_params)
    #     all_results = []
    #     train_label = train[u'label']
    #     weight = np.ones(train_label.shape[0])
    #     for epoch in range(1, epoch_max + 1):
    #         epoch_start_time = time.time()
    #         print('Epoch = {0}/{1}, invoking model.fit'.format(epoch, epoch_max))
    #         history = model.fit(train, test, sample_weight=weight, epoch=1)
    #         training_time = time.time() - epoch_start_time
    #         print('* invoking model.predict')
    #         pred_test = model.predict(test)
    #         print('* invoking ace_eval()')
    #         eval_test = ace_eval(pred_test, test[u'label'], model.num_output,
    #                              num_skipped_events=num_skipped_events, num_total_events=num_total_events)
    #         print('* returned from ace_eval()')
    #         result = {}
    #         result[u'history'] = history
    #         result[u'pred'] = pred_test
    #         result[u'eval'] = eval_test
    #         result[u'description'] = u'epoch={0}'.format(epoch)
    #         if outfile_pattern:
    #             keras_model_filename = outfile_pattern.format(**{u'epoch': epoch, u'type': u'hdf'})
    #             model_filename = outfile_pattern.format(**{u'epoch': epoch, u'type': u'pickle'})
    #             model.description[u'epoch'] = epoch
    #             save_model(model, model_filename, keras_model_filename)
    #             result[u'model_filename'] = model_filename
    #         if epoch < epoch_max:
    #             print(u'Adjusting Weights')
    #             pred_train = model.predict(train)
    #             pred_train_arg = np.argmax(pred_train, axis=1)
    #             for i in range(len(weight)):
    #                 if train_label[i, pred_train_arg[i]] == 1:
    #                     weight[i] /= weight_factor  # I got this correct, so I'm decreasing the weight
    #                 else:
    #                     weight[i] *= weight_factor  # I got this wrong, so I'm increasing the weight
    #             print(u'Adjusting weight max/min={0}/{1}'.format(np.max(weight), np.min(weight)))
    #         epoch_end_time = time.time()
    #         result['train_time'] = training_time
    #         result['epoch_time'] = epoch_end_time - epoch_start_time
    #         all_results.append(result)
    #     return (all_results, model)









