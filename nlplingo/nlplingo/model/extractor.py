
from nlplingo.event.event_domain import EventDomain
from nlplingo.event.event_trigger import EventTriggerGenerator
from nlplingo.event.event_argument import EventArgumentGenerator
from nlplingo.event.event_argument import EventArgumentGeneratorBySentence
from nlplingo.model.trigger_model import CNNTriggerModel
from nlplingo.model.trigger_model import PiecewiseCNNTriggerModel
from nlplingo.model.trigger_model import LSTMTriggerModel
from nlplingo.model.trigger_model import BiLSTMCNNTriggerModel
from nlplingo.model.trigger_model import EmbeddedTriggerModel
from nlplingo.model.trigger_model import OnlineEmbeddedTriggerModel
from nlplingo.model.role_model import MaxPoolEmbeddedRoleModel
from nlplingo.model.role_model import MaxPoolEmbeddedRoleNoTriggerModel
from nlplingo.model.role_model import BidirectionalRoleModel
from nlplingo.model.role_model import BiLSTMMaxPoolEmbeddedRoleModel
from nlplingo.model.role_model import EmbeddedRoleModel

from keras.models import load_model as keras_load_model


class Extractor(object):
    trigger_model_table = {
        'event-trigger_cnn': CNNTriggerModel,
        'event-trigger_piecewise_cnn': PiecewiseCNNTriggerModel,
        'event-trigger_lstm': BiLSTMCNNTriggerModel,
        'event-trigger_lstmcnn': LSTMTriggerModel,
        'event-trigger_embedded': EmbeddedTriggerModel,
        'event-trigger_onlineembedded': OnlineEmbeddedTriggerModel,
    }

    argument_model_table = {
        'event-argument_cnn': MaxPoolEmbeddedRoleModel,
        'event-argument_lstm': BidirectionalRoleModel,
        'event-argument_lstmcnn': BiLSTMMaxPoolEmbeddedRoleModel,
        'event-argument_embedded': EmbeddedRoleModel,
        'event-argument_cnn_no_trigger': MaxPoolEmbeddedRoleNoTriggerModel
    }

    def __init__(self, params, extractor_params, embeddings, load_from_file=False):
        """
        :type params: dict              # general parameters
        :type extractor_params: dict    # specific to this extractor
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        self.model_type = extractor_params['model_type']

        """:type: str"""
        self.domain = EventDomain.read_domain_ontology_file(extractor_params['domain_ontology'],
                                                                 domain_name=extractor_params.get('domain_name', 'general'))
        """:type: nlplingo.event.event_domain.EventDomain"""
        self.model_file = extractor_params['model_file']
        """:type: str"""
        self.hyper_parameters = extractor_params['hyper-parameters']
        """:type: dict"""
        self.model_flags = extractor_params['model_flags']
        """:type: dict"""

        self.generator = None
        self.extraction_model = None
        """:type: nlplingo.model.event_cnn.EventExtractionModel"""

        if self.model_type.startswith('event-trigger_'):
            self.generator = EventTriggerGenerator(
                self.domain,
                params,
                extractor_params['max_sentence_length'],
                extractor_params['hyper-parameters']['neighbor_distance'],
                extractor_params['model_flags'].get('use_bio_index', False)
            )
        elif self.model_type.startswith('event-argument_'):
            self.generator = EventArgumentGenerator(self.domain, params, extractor_params)
            #self.generator = EventArgumentGeneratorBySentence(self.domain, params, extractor_params)
        else:
            raise RuntimeError(
                'Extractor model type: {} not implemented.'.format(self.model_type)
            )
        if len(self.model_file) > 0 and load_from_file:
            self.extraction_model = keras_load_model(self.model_file)
        elif self.model_type in self.trigger_model_table:
            self.extraction_model = self.trigger_model_table[self.model_type](
                extractor_params,
                self.domain,
                embeddings
            )
        elif self.model_type in self.argument_model_table:
            self.extraction_model = self.argument_model_table[self.model_type](
                extractor_params,
                self.domain,
                embeddings
            )
        else:
            raise RuntimeError(
                'Extractor model type: {} not implemented.'.format(self.model_type)
            )
