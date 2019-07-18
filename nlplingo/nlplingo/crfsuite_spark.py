import os
import sys
import re
import json
import time
import zipfile
from argparse import ArgumentParser

from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark import SparkFiles


if __name__ == "__main__":
    conf = SparkConf().setAppName("ner")
    conf.setExecutorEnv('KERAS_BACKEND', 'theano')
    sc = SparkContext(conf=conf)
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default="/user/ychan/data/blog/part-00001-blogs", required=False)
    parser.add_argument("-o", "--output", default="/user/ychan/data/blog/part-00001-blogs.extractions", required=False)
    parser.add_argument("-p", "--partitions", default=20, required=False)
    parser.add_argument("-t", "--type", default="text", required=False)
    args = parser.parse_args()
    num_partitions = int(args.partitions)

    # load parameters
    from nlplingo.common.parameters import Parameters
    params = Parameters('ner.params')
    params.print_params()

    zip_ref = zipfile.ZipFile(params.get_string('resources.zip'), 'r')
    zip_ref.extractall()
    zip_ref.close()


    from nlplingo.ner.ner_feature import NerFeature

    ner_fea = NerFeature(params)

    from nlplingo.ner import decoder
    from nlplingo.ner.decoder import Decoder

    from nlplingo.embeddings.word_embeddings import WordEmbedding
    from nlplingo.event.event_trigger import EventTriggerGenerator
    from nlplingo.event.event_argument import EventArgumentGenerator
    from nlplingo.event.event_domain import CyberDomain

    from nlplingo.event.train_test import generate_trigger_data_feature
    from nlplingo.event.train_test import generate_argument_data_feature
    from nlplingo.event.train_test import load_trigger_model
    from nlplingo.event.train_test import load_argument_model
    from nlplingo.event.train_test import get_predicted_positive_triggers

    word_embeddings = WordEmbedding(params)

    event_domain = None
    if params.get_string('domain') == 'cyber':
        event_domain = CyberDomain()

    arg_generator = EventArgumentGenerator(event_domain, params)
    trigger_generator = EventTriggerGenerator(event_domain, params)

    print('==== Loading Trigger model ====')
    trigger_model = load_trigger_model(params.get_string('event_model_dir'))

    print('==== Loading Argument model ====')
    argument_model = load_argument_model(params.get_string('event_model_dir'))

    #infile = '/user/ychan/data/karma/part-00000'
    #outfile = '/user/ychan/data/out/karma/part-00000'
    #infile_type = 'sequence'

    #infile = '/user/ychan/data/blog/blog.json'
    #outfile = '/user/ychan/data/out/blog/blog.json.extractions'
    #infile_type = 'text'

    #infile = '/user/ychan/data/twitter/tweet.json'
    #outfile = '/user/ychan/data/out/twitter/tweet.json.extractions'
    #infile_type = 'text'

    infile = args.input
    outfile = args.output
    infile_type = args.type

    if infile_type == 'text':
       rdd = sc.textFile(infile).map(lambda x: json.loads(x))\
                                               .map(lambda x: (x["url"], x))\
                                               .repartition(num_partitions) \
                                               .persist(StorageLevel.MEMORY_AND_DISK)
    else:
       rdd = sc.sequenceFile(infile)\
                                               .mapValues(lambda x: json.loads(x))\
                                               .repartition(num_partitions) \
                                               .persist(StorageLevel.MEMORY_AND_DISK)

    print('rdd count %d' % (rdd.count()))

    start_time = time.time()

    def find(element, json):
        return reduce(lambda d, key: d[key], element.split('.'), json)

    def remove_blank_lines(json_data, attribute_name):
        raw_text = find(attribute_name, json_data)
        if type(raw_text) is list:
            clean_text = list()
            for raw_text_item in raw_text:
                clean_text.append(' \n '.join([i.strip() for i in raw_text_item.split('\n') if len(i.strip()) > 0]))
        else:
            clean_text = ' \n '.join([i.strip() for i in raw_text.split('\n') if len(i.strip()) > 0])
        parts = attribute_name.split(".")
        attribute_end_name = parts[len(parts)-1]
        json_end = json_data
        for i in range(0, len(parts)-1):
            json_end = json_end[parts[i]]
        json_end[attribute_end_name] = clean_text

        return json_data

    def apply_cyberlingo(data):
        content_type = None
        attribute_name = None

        if data["source_name"] == 'hg-blogs':
            content_type = "Blog"
            attribute_name = "json_rep.text"
        elif data["source_name"] == 'asu-twitter':
            content_type = "SocialMediaPosting"
            attribute_name = "json_rep.tweetContent"
        elif data["source_name"] == 'isi-news':
            content_type = "NewsArticle"
            attribute_name = "json_rep.readable_text"
        elif data["source_name"] == 'asu-hacking-posts':
            content_type = "Post"
            attribute_name = "json_rep.postContent"

        if content_type is not None:
            data = remove_blank_lines(data, attribute_name)
            return decoder.line_to_predictions(ner_fea, Decoder(params), data, attribute_name, content_type,
                                               word_embeddings, trigger_generator, trigger_model,
                                               arg_generator, argument_model, event_domain)
        return data

    feature_rdd = rdd.mapValues(lambda x : apply_cyberlingo(x))\
                        .mapValues(lambda x: json.dumps(x))\
                        .repartition(10)\
                        .saveAsSequenceFile(outfile)
    
    end_time = time.time()
    print("****************** Elapsed time to save sequence file was %g seconds" % (end_time - start_time))



