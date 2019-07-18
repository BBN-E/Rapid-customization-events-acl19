# This script is used to generate features for both training and decoding.
# The input parameter 'conll_file' points to a file in CoNLL format. Each line consists of 2 space-separated columns. 
# During training, these columns are <word> <NER-tag for word>
# During decoding, these columns are <word> <dummy tag>

import os
import sys
import string
import codecs
from collections import defaultdict

import ner
import crfutils
import json


class NerFeature(object):
    _content_type_to_genre = {
        'NewsArticle': 'news',
        'SocialMediaPosting': 'tweet',
        'Blog': 'blog',
        'Post': 'dw'
    }

    # embeddings : a list of tuples (path-to-embedding-file, float for embedding-scale)
    def __init__(self, params):
        self.params = params

        embeddings = defaultdict(list)
        for content_type in params['content_types']:
            genre = self._content_type_to_genre[content_type]
            for genre_embeddings in params['genres'][genre]['embeddings']:
                embeddings[content_type].append(
                    (
                        genre_embeddings['file'],
                        float(genre_embeddings['scale'])
                    )
                )

        #name_lists_dir = params.get_string('name_lists.dir')
        word_lists = []
        for name_list_entry in params['name_lists']:
            word_lists.append(
                (
                    name_list_entry['label'],
                    name_list_entry['file']
                )
            )

        #brown_cluster_file = os.path.join(params.get_string('brown_cluster.dir'), params.get_string('brown_cluster.file'))
        brown_cluster_file = params['brown_cluster']['file']

        # word embeddings
        self.word_to_embedding = defaultdict(list)
        for content_type, v in embeddings.items():
            for i, (embedding_file, scale) in enumerate(v):
                print('loading embeddings from %s' % (embedding_file))
                self.word_to_embedding[content_type].append({})
                f = codecs.open(embedding_file, 'r', encoding='utf8')
                dimensions = 0
                for line in f:
                    sp = line.strip().split()
                    if len(sp) == 2:	# header information
                        continue
                    dimensions = len(sp) - 1 # should be the same for all lines
                    self.word_to_embedding[content_type][i][sp[0]] = [float(v)*scale for v in sp[1:]]
                self.word_to_embedding[content_type][i]["*UNKNOWN*"] = [0.0 for x in range(dimensions)]

        # brown cluster
        self.brown_clusters = {}
        print('loading brown clusters from %s' % (brown_cluster_file))
        with open(brown_cluster_file) as IN:
            for line in IN:
                fields = line.strip().split()
                self.brown_clusters[fields[0]] = fields[1]

        # Word lists are treated as lowercase
        print('loading word lists')
        self.word_lists = []
        for (label, filename) in word_lists:
            wlist = {}
            with open(filename) as IN:
                for line in IN:
                    name_tokens = line.strip()[1:-1].lower().split()
                    for i,nt in enumerate(name_tokens):
                        if nt not in wlist or i < wlist[nt]:
                            wlist[nt] = i
            self.word_lists.append( (label, wlist) )

        U = [
            'w', 'wl', 'shape', 'shaped', 'type',
            'p1', 'p2', 'p3', 'p4',
            's1', 's2', 's3', 's4',
            '2d', '4d', 'd&a', 'd&-', 'd&/', 'd&,', 'd&.', 'up',
            'iu', 'au', 'al', 'ad', 'ao',
            'cu', 'cl', 'ca', 'cd', 'cs',
        ]
        B = ['w', 'shaped', 'type']

        self.templates = []
        for name in U:
            self.templates += [((name, i),) for i in range(-2, 3)]
        for name in B:
            self.templates += [((name, i), (name, i+1)) for i in range(-2, 2)]

    def get_idf_word_features(self, word, index):
        features = []

        if word.isspace():
            return features

        if word in ["-LRB-", "-RRB-", "(", ")", "[", "]"]:
            features.append("bracket=true")
            return features

        has_digit = False
        has_non_digit = False
        has_punct = False
        has_non_punct = False
        has_alpha = False
        for c in word:
            if c in string.punctuation:
                has_punct = True
            else:
                has_non_punct = True
            if c.isdigit():
                has_digit = True
            else:
                has_non_digit = True
            if c.isalpha():
                has_alpha = True

        if not has_non_punct:
            features.append("punctuation=true")
            return features

        if word.isdigit():
            if len(word) == 2:
                features.append("two-digit-num=true")
            elif len(word) == 4:
                features.append("four-digit-num=true")
            try:
                word_int = int(word)
                if word_int >= 1:
                    if word_int <= 12:
                        features.append("one-to-twelve=true")
                    if word_int <= 31:
                        features.append("one-to-thirty-one=true")
                if word_int >= 0 and word_int <= 60:
                    features.append("zero-to-sixty=true")
                if word_int >= 1900 and word_int <= 2050:
                    features.append("year-num=true")
            except:
                pass
            if len(features) == 0:
                features.append("other-num=true")

        if has_digit:
            if has_alpha:
                features.append("digit-and-alpha=true")
            if word.find("-") >= 0:
                features.append("digit-and-dash=true")
            if word.find("/") >= 0:
                features.append("digit-and-slash=true")
            if word.find(",") >= 0:
                features.append("digit-and-comma=true")
            if word.find(".") >= 0:
                features.append("digit-and-period=true")
            if has_punct:
                features.append("digit-and-punct=true")
        elif has_alpha:
            if word.upper() == word:
                features.append("all-uppercase=true")
            if word.lower() == word:
                features.append("all-lowercase=true")
            if word[0].isupper():
                if len(word) == 2 and word[1] == ".":
                    features.append("cap-period=true")
                elif index == 0:
                    features.append("first-word-init-cap=true")
                else:
                    features.append("init-cap=true")

        if word.find("@") >= 0:
            features.append("has-at-sign=true")

        if has_alpha and has_punct:
            features.append("alpha-and-punct=true")

        if word.startswith('CVE-') >= 0 or word.startswith('cve-') >= 0:
            features.append("has-cve=true")

        #if word.endswith('.apk') or word.endswith('.bmp') or word.endswith('.com') or \
        #   word.endswith('.exe') or word.endswith('.gif') or word.endswith('.html') or \
        #   word.endswith('.inf') or word.endswith('.net') or word.endswith('.php') or \
        #   word.endswith('.sh') or word.endswith('.sys') or word.endswith('.zip'):
        #    features.append("resource-ending=true")

        return features

    def get_brown_prefix(self, word, size):
        if word not in self.brown_clusters:
            return None
        bits = self.brown_clusters[word]
        if len(bits) < size:
            return None # this seems to be what SERIF does
        else:
            return bits[0:size]

    def get_word_list_features(self, word):
        features = []
        word = word.lower()
        if len(word) <= 2:
            return features
        for (label,wlist) in self.word_lists:
            if word in wlist:
                features.append("word-list-%s-%d" % (label, wlist[word]))
        return features


    # seq: a list of tuples, where each tuple is (word, pos-tag, label)
    # for each word in seq, return: label \tab (\tab separated list of features)
    def extract_features(self, seq, content_type):
        ret = []

        genre = self._content_type_to_genre[content_type]
        use_brown_clusters = self.params['genres'][genre]['use_brown_clusters']
        use_traditional_features = self.params['genres'][genre]['use_traditional_features']
        use_idf_word_features = self.params['genres'][genre]['use_idf_word_features']
        use_embeddings = self.params['genres'][genre]['use_embeddings']
        use_lowercase_embeddings = self.params['genres'][genre]['use_lowercase_embeddings']
        use_postag = self.params['genres'][genre]['use_postag']

        if use_traditional_features:
            ner_seq = [{'w': x[0], 'F':[]} for x in seq]
            for x in ner_seq:
                ner.observation(x)
            crfutils.apply_templates(ner_seq, self.templates)
            # ner_seq is the same len as seq. Also, ner_seq[F] is a list of features

        for i in range(2, len(seq)-2):
            fs = []	# list of features for this word
       
            if use_traditional_features:
                fs.extend(ner_seq[i]['F'])

            if use_idf_word_features:
                fs.extend(self.get_idf_word_features(seq[i][0], i-2)) # subtract two to get "real" index
 
            # word features
            #fs.append('U00=%s' % seq[i-2][0])                  # word left-2
            #fs.append('U01=%s' % seq[i-1][0])                  # word left-1
            #fs.append('U02=%s' % seq[i][0])                    # current word (w)
            #fs.append('U03=%s' % seq[i+1][0])                  # word right+1
            #fs.append('U04=%s' % seq[i+2][0])                  # word right+2
            #fs.append('U05=%s/%s' % (seq[i-1][0], seq[i][0]))  # w_left-2 / w
            #fs.append('U06=%s/%s' % (seq[i][0], seq[i+1][0]))  # w / w_right+1

            # lowercase features
            #fs.append('U00lc=%s' % seq[i-2][0].lower())                          # word left-2
            #fs.append('U01lc=%s' % seq[i-1][0].lower())                          # word left-1
            #fs.append('U02lc=%s' % seq[i][0].lower())                            # current word (w)
            #fs.append('U03lc=%s' % seq[i+1][0].lower())                          # word right+1
            #fs.append('U04lc=%s' % seq[i+2][0].lower())                          # word right+2
            #fs.append('U05lc=%s/%s' % (seq[i-1][0].lower(), seq[i][0].lower()))  # w_left-2 / w
            #fs.append('U06lc=%s/%s' % (seq[i][0].lower(), seq[i+1][0].lower()))  # w / w_right+1

            # This will only be non-empty if word lists are specified
            fs.extend(self.get_word_list_features(seq[i][0]))

            if use_brown_clusters:
                # Size is the number of bits in the brown prefix  returned
                for size in [8, 12, 16, 20]:
                    # Get results for five-word window
                    for index in [-2, -1, 0, 1, 2]:
                        bc = self.get_brown_prefix(seq[i+index][0], size)
                        if bc:
                            fs.append("brown.%d.%d=%s" % (size, index, bc))


            # embeddings
            # TODO : try changing the %g to %f
            if use_embeddings:
                for j, embedding in enumerate(self.word_to_embedding[content_type]):
                    # U00e0-0=1:float U00e0-1=1:float U00e1-399=1:float   , word_left-2  embeddings
                    # U01e0-0=1:float ...                                 , word_left-1  embeddings
                    # U02e...                                             , word w       embeddings
                    # U03e...                                             , word_right+1 embeddings
                    # U04e...                                             , word_right+2 embeddings
                    for name, pos in zip(["U00", "U01", "U02", "U03", "U04"], [i-2,i-1,i,i+1,i+2]):
                        w = seq[pos][0]		                # word in that position/index
                        if w not in embedding: w = "*UNKNOWN*"  # default all OOV words to the UNKNOWN embeddings
                        for d in range(len(embedding[w])):
                            fs.append("%se%d-%d=1:%g" % (name, j, d, embedding[w][d]))

            if use_lowercase_embeddings:
                for j, embedding in enumerate(self.word_to_embedding[content_type]):
                    for name, pos in zip(["U00", "U01", "U02", "U03", "U04"], [i-2,i-1,i,i+1,i+2]):
                        w = seq[pos][0]		                # word in that position/index
                        wlc = w.lower()
                        if wlc not in embedding: wlc = "*UNKNOWN*"  # default all OOV words to the UNKNOWN embeddings
                        for d in range(len(embedding[wlc])):
                            fs.append("%slce%d-%d=1:%g" % (name, j, d, embedding[wlc][d]))

            if use_postag:
                fs.append('U10=%s' % seq[i-2][1])
                fs.append('U11=%s' % seq[i-1][1])
                fs.append('U12=%s' % seq[i][1])
                fs.append('U13=%s' % seq[i+1][1])
                fs.append('U14=%s' % seq[i+2][1])
                fs.append('U15=%s/%s' % (seq[i-2][1], seq[i-1][1]))
                fs.append('U16=%s/%s' % (seq[i-1][1], seq[i][1]))
                fs.append('U17=%s/%s' % (seq[i][1], seq[i+1][1]))
                fs.append('U18=%s/%s' % (seq[i+1][1], seq[i+2][1]))
                fs.append('U20=%s/%s/%s' % (seq[i-2][1], seq[i-1][1], seq[i][1]))
                fs.append('U21=%s/%s/%s' % (seq[i-1][1], seq[i][1], seq[i+1][1]))
                fs.append('U22=%s/%s/%s' % (seq[i][1], seq[i+1][1], seq[i+2][1]))


            ret.append("%s\t%s" % (seq[i][2], '\t'.join(fs)))  # example-label , followed by feature vector

        return ret


    def encode(self, x):
        x = x.replace('\\', '\\\\')
        x = x.replace(':', '\\:')
        return x

    # TODO : include postag 
    def line_to_features(self, line):
        d = ('', '', '')
        seq = [d, d]

        for token in line.strip().split(' '):
            seq.append((self.encode(token), 'NN', 'DUMMY-tag'))

        seq.append(d)
        seq.append(d)

        return self.extract_features(seq)



if __name__ == "__main__":

    param_file = sys.argv[3]

    # reading and preparing the parameters
    params = json.load(args.params)
    print(json.dumps(params, sort_keys=True, indent=4))

    content_type = params['content_type']

    ner_feature = NerFeature(params)


    # extracting features
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print('Extracting features\n')
    d = ('', '', '')
    seq = [d, d]
    IN = codecs.open(input_file, 'r', encoding='utf8')
    OUT = codecs.open(output_file, 'w', encoding='utf8')

    for line in IN:
        line = line.strip()
        if len(line) == 0:
            seq.append(d)
            seq.append(d)
            feature_lines = ner_feature.extract_features(seq, content_type)
            for fl in feature_lines:
                OUT.write(fl)
                OUT.write('\n')
            OUT.write('\n')

            seq = [d, d]
        else:
            fields = line.split(' ')
            tag_index = None
            if len(fields) == 2:
                tag_index = 1
            elif len(fields) == 3:
                tag_index = 2
            elif len(fields) == 4:
                tag_index = 3
            if tag_index:
                seq.append((ner_feature.encode(fields[0]), ner_feature.encode(fields[1]), ner_feature.encode(fields[tag_index])))

    IN.close()
    OUT.close()





