from __future__ import absolute_import
from __future__ import division
from __future__ import with_statement

import json

from keras.optimizers import Adadelta, SGD, RMSprop, Adam

global keras_trigger_model
global keras_argument_model


class EventExtractionModel(object):
    verbosity = 0

    keras_custom_objects = {
    #     u'MyRange': MyRange,
    #     u'MySelect': MySelect,
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

        self.word_vec_length = 1                    # because we use word vector index

        if embeddings is not None and 'word_embeddings' in embeddings:
            self.word_embeddings = embeddings['word_embeddings'].word_vec
            """:type: numpy.ndarray"""

        self.optimizer = self._configure_optimizer(params)

        self.data_keys = []
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







