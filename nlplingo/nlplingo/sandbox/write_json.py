import os
import sys
import argparse
import codecs
import json

from nlplingo.common.io_utils import ComplexEncoder

def write_params_json(outfile):
    params = dict()

    extractor_d = dict()
    extractor_d['name'] = 'event-trigger_cnn'
    extractor_d['domain_ontology'] = '/home/ychan/repos/nlplingo/experiments/ace_11172018/role_mappings.ace.txt'
    extractor_d['model_file'] = '/home/ychan/repos/nlplingo/experiments/ace_11172018/trigger.hdf'
    extractor_d['max_sentence_length'] = 100

    hyper_params_d = dict()
    hyper_params_d['positive_weight'] = 3
    hyper_params_d['epoch'] = 10
    hyper_params_d['batch_size'] = 170
    hyper_params_d['number_of_feature_maps'] = 200
    hyper_params_d['neighbor_distance'] = 3
    hyper_params_d['position_embedding_vector_length'] = 5
    hyper_params_d['entity_embedding_vector_length'] = 5
    hyper_params_d['cnn_filter_lengths'] = [3]
    hyper_params_d['dropout'] = 0.5
    extractor_d['hyper-parameters'] = hyper_params_d

    model_flags_d = dict()
    model_flags_d['use_bio_index'] = False
    extractor_d['model_flags'] = model_flags_d

    extractors = []
    extractors.append(extractor_d)
    params['extractors'] = extractors

    params['data'] = dict()
    params['data']['train'] = dict()
    params['data']['train']['filelist'] = '/home/ychan/data/ace/filelist/apf_lingo.laptop.filelist'
    params['data']['train']['features'] = '/home/ychan/repos/nlplingo/experiments/ace_11172018/train.features'

    params['negative_trigger_words'] = '/home/ychan/repos/nlplingo/experiments/ace_11172018/negative_trigger_words'

    params['embeddings'] = dict()
    params['embeddings']['embedding_file'] = '/home/ychan/resources/embeddings/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.withRand.pruned'
    params['embeddings']['vector_size'] = 400
    params['embeddings']['vocab_size'] = 299883
    params['embeddings']['none_token'] = '_pad'
    params['embeddings']['missing_token'] = '_oov'

    params['output_dir'] = '/home/ychan/repos/nlplingo/experiments/ace_11172018'

    print(json.dumps(params, sort_keys=True, indent=4))

    with codecs.open(outfile, 'w', encoding='utf-8') as o:
        o.write(json.dumps(params, sort_keys=True, indent=4, cls=ComplexEncoder, ensure_ascii=False))
        o.write('\n')
        o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    write_params_json(args.params)
