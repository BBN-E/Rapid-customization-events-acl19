from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import os.path
from collections import defaultdict
from io import open

import numpy as np
from future.builtins import range

import config
from ace.ace_docs import AceCorpus
from generator.partition import PartitionGenerator
from generator.trigger import TriggerDataGenerator
from generator.role import RoleDataGenerator
from dataset import Dataset

print("\nace_generate_dataset.py : invoking AceCorpus(word_embedding='baroni')")
# TODO is this needed?
ACE_CORPUS = AceCorpus(word_embedding='baroni')

class AceDataset(Dataset):
    """Top-level for representing trigger/role datasets from ACE dataset."""
    def initialize(self, partition_generator, trigger_generator, role_generator,
                   dataset_name='ace2005', corpus=ACE_CORPUS):
        self.partition_generator = partition_generator
        self.name = dataset_name
        self.trigger_generator = trigger_generator
        self.role_generator = role_generator
        self.corpus = corpus
        self.trigger_data = dict()
        self.role_data = dict()

    def generate(self, dataset_name='ace2005', **params):
        """Generate the datasets"""

        # if using BIO entity annotations, then make sure word embedding has corresponding BIO entity vectors
        if self.trigger_generator.use_bio_index:
            bv = BioVectors(self.corpus, self.partition_generator.train, self.corpus.nlp)
            bv.augment_embedding()

        super(AceDataset, self).generate(dataset_name, **params)
        
    def load(self, dataset_name='ace2005', _quick_debug=False, partition_names=None):
        super(AceDataset, self).load(dataset_name, _quick_debug, partition_names)

    def get_role_subset(self, info_array, partition_name='tst'):
        if not partition_name in self.role_data:
            return self.generate_role_subset(self, info_array)
        
        lookup = defaultdict(list)
        for docid, sent_no, trigger_token_no, trigger_token_i in info_array:
            lookup[docid.decode()].append((sent_no, trigger_token_no))

        role_info = self.role_data[partition_name]['info']
        indices = []
        for idx, (docid, sent_no, trigger_token_no, role_token_no, trigger_token_i, roleTokenI) in np.ndenumerate(role_info):
            if (sent_no, trigger_token_no) in lookup[docid.decode()]:
                indices.append(idx[0])

        result = dict()
        for (key, data) in self.role_data[partition_name].items():
            if len(data.shape) == 0:
                # Probably an object, just copy it
                result[key] = data
            else:
                if len(data.shape) == 1:
                    result[key] = data[indices]
                else:
                    result[key] = data[indices, :]
        return result

    def generate_role_subset(self, info_array):
        self.role_generator.reset()
        previous_docid = None
        for docid, sent_no, trigger_token_no, _ in info_array:
            print(docid, sent_no, trigger_token_no)
            docid = docid.decode()
            if not previous_docid == docid:
                previous_docid = docid
                ace_data = self.corpus.get_document(docid)
            self.role_generator.generate_sentence_trigger_using_mentions(ace_data, sent_no, trigger_token_no)
        return self.role_generator.get_data()

class HengjiPartitionGenerator(PartitionGenerator):
    """Generate the training, development, testing partitions based on Hengji"""
    
    def_dir = config.hengji_def_dir

    def __init__(self, corpus, output_prefix='ace2005-hengji', save=True, _quick_debug=False):

        train_file = os.path.join(self.def_dir, 'train')
        dev_file = os.path.join(self.def_dir, 'dev')
        test_file = os.path.join(self.def_dir, 'test')
        self.train = self.read_docid_path(train_file)
        self.dev = self.read_docid_path(dev_file)
        self.test = self.read_docid_path(test_file)
        self.train = [path.split('/')[-1] for path in self.train]
        self.dev = [path.split('/')[-1] for path in self.dev]
        self.test = [path.split('/')[-1] for path in self.test]

        dataset_properties = dict()
        dataset_properties['partition'] = 'Hengji'

        if _quick_debug:
            # Do a quick run to debug code
            super(HengjiPartitionGenerator, self).__init__(
                corpus,
                [self.test], ['tst'],
                output_prefix=output_prefix, save=save)
        else:
            super(HengjiPartitionGenerator, self).__init__(
                corpus,
                [self.dev, self.train, self.test],
                [ 'dev', 'trn', 'tst'],
                output_prefix=output_prefix, save=save)

    def read_docid_path(self, filename):
        result = []
        with open(filename, 'r') as f:
            for prefix in f:
                docid_path = prefix.strip()
                result.append(docid_path)
        return result

class BioVectors(object):
    """Add entity BIO vectors to word embedding dictionary"""
    def __init__(self, corpus, docid_path_list, nlp):
        self.corpus = corpus
        self.word_embedding = corpus.get_word_embedding()
        self.domain = corpus.domain
        self.docid_path_list = docid_path_list
        self.nlp = nlp
        self.bio_vector_filename = 'bio-vectors-{0}.npz'.format(self.word_embedding.vector_type)
        self.vectors = None

    def process(self):
        self.count = np.zeros(len(self.domain.get_bio_entity_types()),)
        for docid_path in self.docid_path_list:
            ace_data = self.corpus.get_document_by_path(docid_path)
            for sentence in ace_data.sentences:
                for token in sentence:
                    if token.has_vector:
                        if self.vectors is None:
                            self.vectors = np.zeros((len(self.domain.get_bio_entity_types()), len(token.word_vector)))
                        self.count[token.bio_index] += 1
                        self.vectors[token.bio_index, :] += token.word_vector
        for i in range(len(self.domain.get_bio_entity_types())):
            if self.count[i] > 0:
                self.vectors[i, :] /= self.count[i]
            else:
                print('!!!! Bio entity type {0} got zero instances'.format(self.domain.get_bio_entity_types()[i]))

        data_dict = {'vectors': self.vectors, 'count': self.count}
        np.savez(self.bio_vector_filename, **data_dict)

    def load(self):
        if os.path.exists(self.bio_vector_filename):
            result = np.load(self.bio_vector_filename)
            self.vectors = result['vectors']
            self.count = result['count']
        else:
            self.process()

    def augment_embedding(self):
        if not self.vectors:
            self.load()
        should_augment = False
        for i in range(1,len(self.domain.get_bio_entity_types())):
            text, _ , _ = self.word_embedding.get_vector(self.domain.get_bio_entity_types()[i])
            if text is None:
                should_augment = True
                break
        if should_augment:
            for i in range(1,len(self.domain.get_bio_entity_types())):
                # Start with index 1 to skip 'O' of 'BIO'
                self.word_embedding.add_vector(self.domain.get_bio_entity_types()[i], self.vectors[i])
    
def generate_classification_dataset_test():
    prefix = 'ace2005-test'
    pg = HengjiPartitionGenerator(ACE_CORPUS, output_prefix=prefix, save=True, _quick_debug=False)
    
    tg = TriggerDataGenerator(ACE_CORPUS)
    rg = RoleDataGenerator(ACE_CORPUS)
    ad = AceDataset()
    ad.initialize(pg, tg, rg)
    ad.generate(prefix)
    
def generate_classification_dataset(prefix='ace2005-classification', corpus=ACE_CORPUS,
                                  save=True, _quick_debug=False):
    pg = HengjiPartitionGenerator(corpus, output_prefix=prefix, save=save, _quick_debug=_quick_debug)
    # pg = HengjiPartitionGenerator(corpus, output_prefix=prefix, save=True, _quick_debug=True)
    
    tg = TriggerDataGenerator(corpus)
    rg = RoleDataGenerator(corpus)
    ad = AceDataset()
    ad.initialize(pg, tg, rg)
    # ad.generate(prefix, generateTriggerData=False)
    ad.generate(prefix)
    
def generate_classification_dataset_compund_token(prefix='ace2005-classification', corpus=ACE_CORPUS,
                                  save=True, _quick_debug=False):
    pg = HengjiPartitionGenerator(corpus, output_prefix=prefix, save=save, _quick_debug=_quick_debug)
    # pg = HengjiPartitionGenerator(corpus, output_prefix=prefix, save=True, _quick_debug=True)
    
    tg = TriggerDataGenerator(corpus, use_compound_token=True)
    # tg = TriggerDataGenerator(corpus, skip_multi_token_triggers=True, use_word_vector_index=True)
    rg = RoleDataGenerator(corpus)
    ad = AceDataset()
    ad.initialize(pg, tg, rg)
    # ad.generate(prefix, generateTriggerData=False)
    ad.generate(prefix)
    
def generate_identification_dataset():
    prefix = 'ace2005-identification'
    pg = HengjiPartitionGenerator(ACE_CORPUS, output_prefix=prefix, save=True, _quick_debug=False)
    tg = TriggerDataGenerator(ACE_CORPUS, generate_identification=True)
    rg = RoleDataGenerator(ACE_CORPUS, generate_identification=True)
    ad = AceDataset()
    ad.initialize(pg, tg, rg)
    ad.generate(prefix)

if __name__ == "__main__":
    generate_classification_dataset(corpus=ACE_CORPUS)
    # generateClassificationDataset(prefix='ace2005-classification-google', corpus=AceCorpus(word_embedding='google'))
    # generateClassificationDataset(prefix='ace2005-classification-googleNYT', corpus=AceCorpus(word_embedding='googleNYT'))
    # generateClassificationDatasetCompundToken(prefix='ace2005-googleNYT-compoundToken', corpus=AceCorpus(word_embedding='googleNYT'))
