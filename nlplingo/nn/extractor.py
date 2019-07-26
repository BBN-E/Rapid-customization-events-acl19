from keras.models import load_model as keras_load_model
from nlplingo.event.argument.feature import EventArgumentFeatureGenerator
from nlplingo.event.argument.generator import EventArgumentExampleGenerator
from nlplingo.event.event_domain import EventDomain
from nlplingo.event.trigger.feature import EventTriggerFeatureGenerator
from nlplingo.event.trigger.generator import EventTriggerExampleGenerator

from nlplingo.nn.argument_model import CNNArgumentModel
from nlplingo.nn.trigger_model import CNNTriggerModel


class HyperParameters(object):
    def __init__(self, params, load_from_file=False):
        if load_from_file:
            self.max_sentence_length = params.get('max_sentence_length')
            self.neighbor_distance = params.get('neighbor_distance')
        else:
            self.positive_weight = params['positive_weight']
            self.epoch = params['epoch']
            self.early_stopping = params.get('early_stopping', False)
            self.number_of_feature_maps = params.get('number_of_feature_maps', 0)  # number of convolution feature maps
            self.batch_size = params['batch_size']

            self.position_embedding_vector_length = params['position_embedding_vector_length']
            self.entity_embedding_vector_length = params['entity_embedding_vector_length']
            self.filter_lengths = params.get('cnn_filter_lengths', 0)  # list[int]
            self.dropout = params['dropout']
            #self.use_event_embedding = params.get('use_event_embedding', True)
            self.max_sentence_length = params.get('max_sentence_length')
            self.neighbor_distance = params.get('neighbor_distance')

            self.train_embeddings = params.get('train_embeddings')
            self.finetune_epoch = params.get('fine-tune_epoch')


class Extractor(object):
    trigger_model_table = {
        'event-trigger_cnn': CNNTriggerModel
    }

    argument_model_table = {
        'event-argument_cnn': CNNArgumentModel
    }

    def __init__(self, params, extractor_params, embeddings, load_from_file=False):
        """
        :type params: dict              # general parameters
        :type extractor_params: dict    # specific to this extractor
        :type embeddings: dict[str : nlplingo.embeddings.word_embeddings.WordEmbedding]
        """
        self.model_type = extractor_params['model_type']
        """:type: str"""

        self.domain = EventDomain.read_domain_ontology_file(extractor_params['domain_ontology'],
                                                            domain_name=extractor_params.get('domain_name', 'general'))
        """:type: nlplingo.event.event_domain.EventDomain"""

        self.model_file = extractor_params['model_file']
        """:type: str"""

        self.hyper_parameters = HyperParameters(extractor_params['hyper-parameters'], load_from_file)
        """:type: nlplingo.nn.extractor.HyperParameters"""

        self.feature_generator = None  # feature generator
        self.example_generator = None  # example generator

        if self.model_type.startswith('event-trigger_'):
            self.feature_generator = EventTriggerFeatureGenerator(extractor_params)
            self.example_generator = EventTriggerExampleGenerator(self.domain, params, extractor_params,
                                                                  self.hyper_parameters)
        elif self.model_type.startswith('event-argument_'):
            self.feature_generator = EventArgumentFeatureGenerator(extractor_params)
            self.example_generator = EventArgumentExampleGenerator(self.domain, params, extractor_params,
                                                                   self.hyper_parameters)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(self.model_type))

        self.extraction_model = None
        """:type: nlplingo.nn.event_model.EventExtractionModel"""
        if load_from_file:
            print('Loading previously trained model')
            self.extraction_model = keras_load_model(self.model_file)
        elif self.model_type in self.trigger_model_table:
            self.extraction_model = self.trigger_model_table[self.model_type](extractor_params, self.domain, embeddings,
                                                                              self.hyper_parameters,
                                                                              self.feature_generator.features)
        elif self.model_type in self.argument_model_table:
            self.extraction_model = self.argument_model_table[self.model_type](extractor_params, self.domain,
                                                                               embeddings, self.hyper_parameters,
                                                                               self.feature_generator.features)
        else:
            raise RuntimeError('Extractor model type: {} not implemented.'.format(self.model_type))
