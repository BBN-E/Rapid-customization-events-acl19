
import os
import sys
import re
import json
import time
import zipfile

from pyspark import SparkContext, SparkConf
#from ctypes import *
from pyspark import SparkFiles

class Parameters(object):

    def __init__(self, config_file):
        self.params = {}

        f = open(config_file, 'r')
        for line in f:
            line = line.strip()

            if len(line) == 0 or line.startswith('#'):
                continue

            index = line.find(':')
            key = line[0:index].strip()
            value = line[index+1:].strip()

            match_obj = re.match(r'%(.*?)%', value)
            if match_obj:
                re_key = match_obj.group(1)
                if re_key in self.params:
                    re_value = self.params[re_key]
                    value = re.sub(r'%(.*?)%', re_value, value)

            self.params[key] = value

        f.close()

    def get_string(self, key):
        return self.params.get(key)

    def get_list(self, key):
        if key in self.params:
            return self.params.get(key).split(',')
        else:
            return None

    def get_boolean(self, key):
        if key in self.params:
            return self.to_boolean(self.params.get(key))
        else:
            return None

    def to_boolean(self, v):
        if v == 'True' or v == 'true':
            return True
        else:
            return False

    def print_params(self):
        for k, v in self.params.items():
            print('%s: %s' % (k, v))


if __name__ == "__main__":
    conf = SparkConf().setAppName("ner")
    sc = SparkContext(conf=conf)

    #print('**** %s' % (os.listdir(SparkFiles.get(''))))

    # load parameters
    #param_file = sys.argv[1]
    params = Parameters('ner.params')
    params.print_params()

    zip_ref = zipfile.ZipFile(params.get_string('resources.zip'), 'r')
    zip_ref.extractall()
    zip_ref.close()

    #crfsuite_python = params.get_string('crfsuite.python')
    #crfsuite_lib = params.get_string('crfsuite.lib')
#    model_blog = params.get_string('model.blog')
#    model_tweet = params.get_string('model.tweet')
    #base_dir = params.get_string("base_dir")
    #ner_dir = params.get_string("ner_dir")

    # add paths and modules
    #sys.path.append(crfsuite_python)
    #crfsuite_libs = [ner_dir+'/crfsuite/liblbfgs-1.10.so', ner_dir+'/crfsuite/libcqdb-0.12.so', ner_dir+'/crfsuite/libcrfsuite-0.12.so']
    #for library in crfsuite_libs:
    #    cdll.LoadLibrary(library)
    #sc.addPyFile(crfsuite_python + '/ner_feature.py')
    #sc.addPyFile(crfsuite_python + '/ner.py')
    #sc.addPyFile(crfsuite_python + '/crfutils.py')
    #sc.addPyFile(crfsuite_python + '/parameters.py')
    #sc.addPyFile(crfsuite_python + '/decoder.py')
    #sc.addPyFile(crfsuite_python + '/emoticons.py')
    #sc.addPyFile(crfsuite_python + '/twokenize.py')

    from bbn.ner_feature import NerFeature

    ner_fea = NerFeature(params)

    from bbn import decoder
    from bbn.decoder import Decoder

    attribute_name = 'text'

    #infile = '/user/ychan/data/karma/part-00000'
    #outfile = '/user/ychan/data/out/karma/part-00000'
    #infile_type = 'sequence'

    infile = '/user/ychan/data/blog/blog.json'
    outfile = '/user/ychan/data/out/blog/blog.json.extractions'
    infile_type = 'text'

    #infile = '/user/ychan/data/twitter/tweet.json'
    #outfile = '/user/ychan/data/out/twitter/tweet.json.extractions'
    #infile_type = 'text'


    if infile_type == 'text':
       rdd = sc.textFile(infile).map(lambda x: json.loads(x)).map(lambda x: (x["url"], x))
    else:
       rdd = sc.sequenceFile(infile).mapValues(lambda x: json.loads(x))

    print('rdd count %d' % (rdd.count()))


    start_time = time.time()
    feature_rdd = rdd.mapValues(lambda x : decoder.line_to_predictions(ner_fea, Decoder(params), x, attribute_name))
    #for fv in feature_rdd.take(3):
    #    print(fv)
    #end_time = time.time()
    #print("****************** Elapsed time to transform RDD was %g seconds" % (end_time - start_time))

    #start_time = time.time()
    feature_rdd.mapValues(lambda x: json.dumps(x)).saveAsSequenceFile(outfile)
    end_time = time.time()
    print("****************** Elapsed time to save sequence file was %g seconds" % (end_time - start_time))



