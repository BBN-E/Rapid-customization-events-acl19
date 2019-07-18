
import os
import argparse
import codecs
import re
import glob
import json

from collections import defaultdict

class Token(object):
    def __init__(self, text):
        """
        :type text: str
        :type index: int
        """
        self.text = text
        self.index = None  # index of the token within the sentence
        self.lexsn = None
        self.wnsn = None
        self.lemma = None
        self.pos = None     # pos-tag

class Sentence(object):
    def __init__(self, tokens):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :param tokens:
        """
        self.tokens = tokens
        """:type: list[nlplingo..text.text_span.Token]"""

class Document(object):
    def __init__(self, docid):
        self.docid = docid
        self.sentences = []
        """:type: list[nlplingo.text.text_span.Sentence]"""

    def add_sentence(self, sentence):
        """
        :type sentence: list[nlplingo.text.text_span.Sentence]
        """
        self.sentences.append(sentence)

def read_semcor_document(filepath):
    docid = os.path.basename(filepath)
    doc = Document(docid)

    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<s '):
                tokens = []
            if line.startswith('<wf ') or line.startswith('<punc>'):
                tokens.append(line)
            if line.startswith('</s>'):
                sentence = xmltokens_to_sentence(tokens)
                doc.add_sentence(sentence)
    return doc

def xmltoken_to_token(xmltoken):
    search_obj = re.search(r'^<(.*)>(.*?)</', xmltoken)
    xmlstring = search_obj.group(1)
    text = search_obj.group(2)

    token = Token(text)
    for t in xmlstring.split():
        if t.startswith('pos='):
            token.pos = t[4:]
        if t.startswith('lemma='):
            token.lemma = t[6:]
        if t.startswith('wnsn='):
            token.wnsn = t[5:]
        if t.startswith('lexsn='):
            token.lexsn = t[6:]
    return token

def xmltokens_to_sentence(xmltokens):
    tokens = []
    for i, xmltoken in enumerate(xmltokens):
        token = xmltoken_to_token(xmltoken)
        token.index = i
        tokens.append(token)
    return Sentence(tokens)

class JsonObject:
    # Serialization helper
    def reprJSON(self):
        d = dict()
        for a, v in self.__dict__.items():
            if v is None:
                continue
            if (hasattr(v, "reprJSON")):
                d[a] = v.reprJSON()
            else:
                d[a] = v
        return d


class SenseExample(JsonObject):
    def __init__(self, lemma, lexsn, wnsn):
        self.lemma = lemma
        self.lexsn = lexsn
        self.wnsn = wnsn
        self.start_token_index = None
        self.end_token_index = None
        self.target_sentence = None     # space separated text
        self.previous_sentence = None   # space separated text
        self.next_sentence = None       # space separated text

class LemmaSenseData(JsonObject):
    def __init__(self, lemma):
        self.lemma = lemma
        self.sense_data = defaultdict(list) # mapping from wnsn to list[SenseExample]

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)

# python nlp/semcor.py --datadir /home/ychan/data/semcor3.0/brown1/tagfiles,/home/ychan/data/semcor3.0/brown2/tagfiles,/home/ychan/data/semcor3.0/brownv/tagfiles --outdir /home/ychan/data/semcor3.0_json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')    # dir containing semcor files
    parser.add_argument('--outdir')
    args = parser.parse_args()

    sense_data = dict()
    """:type: dict(str, LemmaSenseData)"""

    for dirpath in args.datadir.split(','):
        for filepath in glob.glob(dirpath+'/*'):
            doc = read_semcor_document(filepath)

            for sent_index, sentence in enumerate(doc.sentences):
                for token in sentence.tokens:
                    if token.lexsn is not None:
                        eg = SenseExample(token.lemma, token.lexsn, token.wnsn)
                        eg.start_token_index = token.index
                        eg.end_token_index = token.index
                        eg.target_sentence = ' '.join(t.text for t in sentence.tokens)
                        if sent_index > 0:
                            eg.previous_sentence = ' '.join(t.text for t in doc.sentences[sent_index - 1].tokens)
                        else:
                            eg.previous_sentence = None
                        if (sent_index + 1) < len(doc.sentences):
                            eg.next_sentence = ' '.join(t.text for t in doc.sentences[sent_index + 1].tokens)
                        else:
                            eg.next_sentence = None

                        lemma_string = None
                        if token.pos.startswith('N'):
                            lemma_string = token.lemma+'.n'
                        elif token.pos.startswith('V'):
                            lemma_string = token.lemma+'.v'

                        if lemma_string is None:
                            continue

                        if lemma_string in sense_data:
                            sense_data[lemma_string].sense_data[token.wnsn].append(eg)
                        else:
                            d = LemmaSenseData(token.lemma)
                            d.sense_data[token.wnsn].append(eg)
                            sense_data[lemma_string] = d

    stats = defaultdict(int)
    for lemma_string, lemma_data in sense_data.items():
        # calculate some statistics
        data = lemma_data.sense_data
        eg_count = 0
        for sense in data.keys():
            eg_count += len(data[sense])
        if '_' in lemma_string:
            stats['ngram_lemma_count'] += 1
            stats['ngram_sense_count'] += len(data)
            stats['ngram_eg_count'] += eg_count
            if lemma_string.endswith('.n'):
                stats['ngram_noun'] += 1
            elif lemma_string.endswith('.v'):
                stats['ngram_verb'] += 1
        else:
            stats['unigram_lemma_count'] += 1
            stats['unigram_sense_count'] += len(data)
            stats['unigram_eg_count'] += eg_count
            if lemma_string.endswith('.n'):
                stats['unigram_noun'] += 1
            elif lemma_string.endswith('.v'):
                stats['unigram_verb'] += 1

        #print('{} has {} senses in {} examples'.format(lemma_string, len(data), eg_count))

        with codecs.open(args.outdir + '/' + lemma_string, 'w', encoding='utf-8') as o:
            o.write(json.dumps(lemma_data, sort_keys=True, indent=4, cls=ComplexEncoder, ensure_ascii=False))
            o.close()

    print('{} unigrams'.format(stats['unigram_lemma_count']))
    print('- {} nouns, {} verbs'.format(stats['unigram_noun'], stats['unigram_verb']))
    print('- {} senses, {} examples'.format(stats['unigram_sense_count'], stats['unigram_eg_count']))
    print('- AVG per word, %.1f senses, %.1f examples' % (float(stats['unigram_sense_count']) / stats['unigram_lemma_count'],
                                                          float(stats['unigram_eg_count']) / stats['unigram_lemma_count']))

    print('{} ngrams'.format(stats['ngram_lemma_count']))
    print('- {} nouns, {} verbs'.format(stats['ngram_noun'], stats['ngram_verb']))
    print('- {} senses, {} examples'.format(stats['ngram_sense_count'], stats['ngram_eg_count']))
    print('- AVG per word, %.1f senses, %.1f examples' % (float(stats['ngram_sense_count']) / stats['ngram_lemma_count'],
                                                    float(stats['ngram_eg_count']) / stats['ngram_lemma_count']))
