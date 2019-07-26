import sys
import codecs
import json

from nlplingo.common.utils import IntPair
from nlplingo.text.text_span import TextSpan
from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import SRL


class JSONTokens(object):
    def __init__(self, text):
        self.text = text
        self.index = -1
        self.srl_predicate = None
        self.srl_roles = None

    def to_string(self):
        if self.srl_predicate is not None:
            srl_string = '{}|||{}'.format(self.srl_predicate, ' '.join('{}:{}'.format(k, v) for k,v in self.srl_roles.items()))
        else:
            srl_string = ''
        return '{}:{} {}'.format(self.index, self.text, srl_string)


def _convert_to_tokens(token_sentence, srl_indices):
    """
    :type token_strings: list[str]
    Returns: list[ColumnToken]
    """
    ret = []

    tokens = []
    """:type: list[list[str]]"""
    for token in token_sentence:
        tokens.append(token)

    predicates = dict()
    #arguments = {}
    for item in srl_indices:
        if item[0] not in predicates:
            predicates[item[0]] = []
        predicates[item[0]].append([item[1], item[2], item[3].replace("RG", "")])
        """
        predicates.append(item[0])
        start_index = item[1]
        end_index = item[2]
        for i in range(start_index, end_index):
            arguments[i] = item[3].replace("RG", "")
        """


    predicate_count = 0

    for index, token in enumerate(tokens):
        word = token
        #lprint(word)

        ct = JSONTokens(word)
        ct.index = index

        if index in predicates:
            predicate_count += 1
            srl_roles = dict()
            if index not in srl_roles:
                srl_roles[index] = []
            srl_roles[index] = predicates[index]
            ct.srl_predicate = token
            ct.srl_roles = srl_roles
        ret.append(ct)

    return ret


def add_srl_annotations(doc, srl_filepath, offset_filepath):
    """
    :type doc: nlplingo.text.text_theory.Document
    """

    sentences = []
    """list[nlplingo.text.text_span.Sentence]"""
    with codecs.open(offset_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            offset = IntPair(int(tokens[0]), int(tokens[1]))
            # now, let's find the Sentence object with this offset
            sentence_match = None
            for sentence in doc.sentences:
                if offset.first == sentence.start_char_offset() and offset.second == sentence.end_char_offset():
                    sentence_match = sentence
                    break
            assert sentence_match is not None
            sentences.append(sentence_match)

    srl_sentences = []
    """:type: list[list[ColumnToken]]"""
    token_strings = []


    with codecs.open(srl_filepath, 'r', encoding='utf-8') as f:

        srl_json = json.load(f)

        srl_token_strings = srl_json["sentences"]
        srl_tags = srl_json["predicted_srl"]

        srl_indices = []

        threshold_length = 0

        for line in srl_token_strings:
            srl_sentence_split = line
            srl_sentence_indices = len(line)
            srl_indices = []
            for item in srl_tags:

                # offsets are fixed

                if int(item[0]) >= threshold_length and int(item[1]) >= threshold_length and int(item[2]) >= threshold_length:
                    if int(item[0]) < threshold_length+len(line) and int(item[1]) < threshold_length+len(line) and int(item[2]) < threshold_length+len(line):
                        srl_indices.append([int(item[0])-threshold_length, int(item[1])-threshold_length, int(item[2])-threshold_length, item[3]])

            threshold_length = threshold_length + len(line)
            #print(line)
            #print(srl_indices)
            srl_sentences.append(_convert_to_tokens(line, srl_indices))

            """   
            line = line.strip()
            if len(line) == 0:
                if len(token_strings) > 0:
                    srl_sentences.append(_convert_to_tokens(token_strings))
                    token_strings = []
            else:
                token_strings.append(line)
            """
    assert len(sentences) == len(srl_sentences), 'len(sentences)={} len(srl_sentences)={}'.format(str(len(sentences)), str(len(srl_sentences)))
    #for i, sentence in enumerate(sentences):
    #    assert len(srl_sentences[i]) == len(sentence.tokens), 'i={} start={} end={}'.format(str(
    # i), str(sentence.start_char_offset()), str(sentence.end_char_offset()))

    for sentence_index, srl_sentence in enumerate(srl_sentences):
        sentence = sentences[sentence_index]
        """:type: nlplingo.text.text_span.Sentence"""

        if len(srl_sentence) != len(sentence.tokens):
            srl_tokens_string = ' '.join(t.text for t in srl_sentence)
            sentence_tokens_string = ' '.join(t.text for t in sentence.tokens)
            print('add_srl_annotation: Skipping doc {} sentence {}: len(srl_sentence)={} '
                  'len(sentence.tokens)={}'.format(doc.docid, str(sentence_index),
                                                   str(len(srl_sentence)), str(len(sentence.tokens))))
            print(' - sen_tokens: {}'.format(sentence_tokens_string.encode("utf-8")))
            print(' - srl_tokens: {}'.format(srl_tokens_string.encode("utf-8")))
            continue

        for column_token_index, column_token in enumerate(srl_sentence):

            if column_token.srl_predicate is not None:
                srl = SRL('dummy')
                srl.predicate_label = column_token.srl_predicate
                #print(column_token.srl_roles)
                for srl_role in column_token.srl_roles[column_token_index]:
                    #if token_index != column_token.srl_roles:   # omit role-arguments that are also the predicate
                    #print(column_token_index)
                    #print(srl_role)
                    srl.add_role(srl_role[2], srl_role[0], srl_role[1])

                #print(srl.roles['A0'])
                # expand 'A0' srl-role to its compound and appos
                #print(srl.roles['A0'])
                for token_index1, token_index2 in srl.roles['A0']:
                    token = sentence.tokens[token_index1]
                    expanded_indices = set(r.connecting_token_index for r in token.dep_relations if 'compound' in r.dep_name)
                    #print(expanded_indices)
                    for i in expanded_indices:
                        #if i != column_token.index and i != token_index:
                        srl.add_role('A0:compound', i, token_index2)
                sentence.tokens[column_token_index].srl = srl


