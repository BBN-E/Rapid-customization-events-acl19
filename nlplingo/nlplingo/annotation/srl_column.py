import sys
import codecs

from collections import defaultdict

from nlplingo.common.utils import IntPair
from nlplingo.text.text_span import TextSpan
from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import SRL

class ColumnToken(object):
    def __init__(self, text):
        self.text = text
        self.lemma = None
        self.pos_tag = None
        self.index = -1
        self.srl_predicate = None
        self.srl_roles = None

    def to_string(self):
        if self.srl_predicate is not None:
            srl_string = '{}|||{}'.format(self.srl_predicate, ' '.join('{}:{}'.format(k, v) for k,v in self.srl_roles.items()))
        else:
            srl_string = ''
        return '{}:{}/{}/{} {}'.format(self.index, self.text, self.lemma, self.pos_tag, srl_string)


def _convert_to_tokens(token_strings):
    """
    :type token_strings: list[str]
    Returns: list[ColumnToken]
    """
    ret = []

    tokens = []
    """:type: list[list[str]]"""
    for token_string in token_strings:
        tokens.append(token_string.split('\t'))

    predicate_count = 0
    for index, token in enumerate(tokens):
        word = token[1]
        lemma = token[2]
        pos_tag = token[4]

        ct = ColumnToken(word)
        ct.lemma = lemma
        ct.pos_tag = pos_tag
        ct.index = index

        predicate = token[13]
        if predicate != '_':
            predicate_count += 1
            srl_roles = dict()
            for i, t in enumerate(tokens):
                role = t[13 + predicate_count]
                if role != '_':
                    srl_roles[i] = role
            ct.srl_predicate = predicate
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
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(token_strings) > 0:
                    srl_sentences.append(_convert_to_tokens(token_strings))
                    token_strings = []
            else:
                token_strings.append(line)

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
            print(' - sen_tokens: {}'.format(sentence_tokens_string))
            print(' - srl_tokens: {}'.format(srl_tokens_string))
            continue

        for column_token_index, column_token in enumerate(srl_sentence):
            if column_token.srl_predicate is not None:
                srl = SRL('dummy')
                srl.predicate_label = column_token.srl_predicate
                for token_index, srl_role in column_token.srl_roles.items():
                    if token_index != column_token.index:   # omit role-arguments that are also the predicate
                        srl.add_role(srl_role, token_index)

                # expand 'A0' srl-role to its compound and appos
                for token_index in srl.roles['A0']:
                    token = sentence.tokens[token_index]
                    expanded_indices = set(r.connecting_token_index for r in token.dep_relations if 'compound' in r.dep_name)
                    for i in expanded_indices:
                        if i != column_token.index and i != token_index:
                            srl.add_role('A0:compound', i)

                sentence.tokens[column_token_index].srl = srl



# def process_srl_file(doc, filepath):
#     """Reads PathLSTM annotations from filename, and add to doc
#
#     :type filepath: str
#     :type doc: nlplingo.text.text_theory.Document
#     """
#
#     #print('Processing SRL file {}'.format(filepath))
#
#     srl_sentences = []
#     """:type: list[list[ColumnToken]]"""
#     token_strings = []
#     with codecs.open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if len(line) == 0:
#                 if len(token_strings) > 0:
#                     srl_sentences.append(convert_to_tokens(token_strings))
#                     token_strings = []
#             else:
#                 token_strings.append(line)
#
#     #assert len(doc.sentences) == len(sentences)
#
#     for sentence_index, sentence in enumerate(srl_sentences):
#         st = doc.sentences[sentence_index]
#         assert len(sentence) == len(st.tokens)
#         for column_token in sentence:
#             if column_token.srl_predicate is not None:
#                 srl = SRL('dummy')
#                 srl.predicate_label = column_token.srl_predicate
#                 srl.predicate_token = st.tokens[column_token.index]
#                 for k, v in column_token.srl_roles.items():
#                     token = st.tokens[k]
#                     arg_span = TextSpan(IntPair(token.start_char_offset(), token.end_char_offset()), token.text)
#                     arg_span.with_tokens([token])
#                     if k != column_token.index:
#                         srl.add_role(v, arg_span)
#
#
#                 # expand 'A0' srl-role to its compound and appos
#                 new_roles = defaultdict(list)
#                 for srl_role in srl.roles:
#                     if srl_role == 'A0':
#                         for span in srl.roles[srl_role]:
#                             token = span.tokens[0]
#
#                             token_indices = set()
#                             for dep_rel in token.dep_relations + token.child_dep_relations:
#                                 if 'compound' in dep_rel.dep_name or 'appos' in dep_rel.dep_name:
#                                     token_indices.add(dep_rel.child_token_index)
#                                     token_indices.add(dep_rel.parent_token_index)
#
#                             expanded_indices = set(index for index in token_indices if (index != token.index_in_sentence and index != srl.predicate_token.index_in_sentence))
#                             for i in expanded_indices:
#                                 t = st.tokens[i]
#                                 arg_span = TextSpan(IntPair(t.start_char_offset(), t.end_char_offset()), t.text)
#                                 new_roles['A0:compound'].append(arg_span)
#                 for r in new_roles:
#                     for span in new_roles[r]:
#                         srl.add_role(r, span)
#
#                 st.add_srl(srl)
#
#     return doc


if __name__ == "__main__":
    text_file = sys.argv[2]
    srl_file = sys.argv[1]

    with codecs.open(text_file, 'r', encoding='utf-8') as f:
        doc = Document('dummy', text=f.read())

    process_srl_file(doc, srl_file)


