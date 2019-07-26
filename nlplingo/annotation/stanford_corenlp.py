from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import json

import argparse
import codecs
import re

from collections import defaultdict

from nlplingo.common import utils
from nlplingo.common.utils import IntPair
from nlplingo.common.span_utils import offset_overlap
from nlplingo.common.span_utils import offset_same
from nlplingo.text.text_span import EventSpan
from nlplingo.text.text_theory import Event
from nlplingo.text.text_span import EventArgument
from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import TextSpan
from nlplingo.text.text_span import Token
from nlplingo.text.dependency_relation import DependencyRelation
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_theory import Document


def _read_sentence_tokens(sentence_json):
    """
    Returns: list[nlplingo.text.text_span.Token]
    """
    ret = []
    for i, token in enumerate(sentence_json['tokens']):
        word = token['originalText']
        lemma = token['lemma']
        start = token['characterOffsetBegin']
        end = token['characterOffsetEnd']
        pos_tag = token['pos']
        ner = token['ner']

        ret.append(Token(IntPair(start, end), i, word, lemma, pos_tag))
    return ret

def _read_sentence_dependency_parse(sentence_json, sentence_tokens):
    """
    :type sentence_tokens: list[nlplingo.text.text_span.Token]
    """
    for dep_json in sentence_json['enhancedPlusPlusDependencies']:
        child_token_index = dep_json['dependent'] - 1
        parent_token_index = dep_json['governor'] - 1
        relation_name = dep_json['dep']

        sentence_tokens[child_token_index].add_dep_relation(
            DependencyRelation(relation_name, DependencyRelation.up(), parent_token_index))
        sentence_tokens[parent_token_index].add_dep_relation(
            DependencyRelation(relation_name, DependencyRelation.down(), child_token_index))

        #r = DependencyRelation(relation_name, parent_token_index, child_token_index)
        #sentence_tokens[child_token_index].add_dep_relation(r)
        #sentence_tokens[parent_token_index].add_child_dep_relation(r)

def _read_sentence(sentence_json):
    """
    Returns: (int, list[nlplingo.text.text_span.Token])
    """
    index = sentence_json['index']                          # sentence index
    tokens = _read_sentence_tokens(sentence_json)            # construct Token objects for sentence
    _read_sentence_dependency_parse(sentence_json, tokens)   # add dependency annotations to tokens
    return (index, tokens)


def find_trigger_candidates(target_token, sentence, words_to_skip, verb_words):
    """
    :type target_token: nlplingo.text.text_span.Token
    :type sentence: nlplingo.text.text_span.Sentence
    :type words_to_skip: set(str)

    Returns: dict(int, list[str])
    int represents token index of suggested trigger, str represents dependency relation
    """
    candidates = defaultdict(list)


    tokens = [token for token in sentence.tokens if ((
                                                     token.pos_category() == 'NOUN' or token.pos_category() == 'VERB') and token.index_in_sentence != target_token.index_in_sentence)]

    for token in tokens:
        paths = find_shortest_dep_paths_between_tokens(target_token, token, sentence.tokens)

        for path in paths:
            ##print('Examining: {}'.format(' '.join(r for r in path)))

            i = 0
            if path[i].startswith('u:nmod:within:') or path[i].startswith('u:nmod:in:'):
                token_index = int(path[i].split(':')[-2])
                if sentence.tokens[token_index].text.lower() in verb_words:
                    candidates[token_index].append(path[i])
                    #candidates.add('{}:{}'.format(i, path[i]))
                continue

            while (i < len(path) and (
                    path[i].startswith('u:appos') or path[i].startswith('d:appos') or \
                    path[i].startswith('u:amod') or path[i].startswith('d:amod') or \
                    path[i].startswith('u:compound') or path[i].startswith('d:compound') or \
                    path[i].startswith('u:dep:') or path[i].startswith('d:dep:') or \
                    path[i].startswith('u:nsubjpass') or path[i].startswith('d:nsubjpass') or \
                    (path[i].startswith('u:nmod') and not path[i].startswith('u:nmod:agent')) or \
                    (path[i].startswith('d:nmod') and not path[i].startswith('d:nmod:agent')) or \

                    path[i].split(':')[-1].lower() in words_to_skip)):
                i += 1

            if i < len(path):
                while i < len(path) and (path[i].startswith('u:nsubj') or path[i].startswith('u:acl') or \
                        path[i].startswith('d:acl') or path[i].startswith('d:xcomp') or path[i].startswith('d:ccomp') or \
                        path[i].startswith('u:nmod:agent') or path[i].startswith('d:nmod:agent')):
                        #path[i].startswith('d:dobj'):
                    strings = path[i].split(':')
                    #trigger_text = strings[-1]
                    trigger_token_index = int(strings[-2])
                    if path[i].split(':')[-1].lower() not in words_to_skip:
                        candidates[trigger_token_index].append(path[i])
                        #candidates.add('{}:{}'.format(i, path[i]))
                    i += 1

                # if i < len(path):
                #     if path[i].startswith('d:dobj:') or path[i].startswith('d:nmod:for:'):
                #         token_index = int(path[i].split(':')[-2])
                #         if sentence.tokens[token_index].text.lower() in verb_words:
                #             candidates.add('{}:{}'.format(i, path[i]))

    return candidates

    candidates_by_deptype = defaultdict(list)
    for cand in candidates:
        parts = cand.split(':')
        dep_prefix = ':'.join(parts[0:-2])
        candidates_by_deptype[dep_prefix].append(':'.join(parts[-2:]))
    print('len(candidates_by_deptype)={}'.format(len(candidates_by_deptype)))

    ret = []
    # select max of 1 candidate for each dependency prefix
    for dep_prefix in candidates_by_deptype:
        cands = candidates_by_deptype[dep_prefix]
        if len(cands) == 1:
            ret.append('{}:{}'.format(dep_prefix, cands[0]))
        else:
            min_distance = 99
            best_cand = None
            for cand in cands:
                index = int(cand.split(':')[0])
                distance = abs(index - target_token.index_in_sentence)
                if distance < min_distance:
                    min_distance = distance
                    best_cand = cand
            ret.append('{}:{}'.format(dep_prefix, best_cand))
    print('len(ret)={}'.format(len(ret)))
    return ret

    # nsubj = []
    # nsubjXsubj = []
    # for cand in candidates:
    #     if 'u:nsubj' in cand:
    #         if 'u:nsubj:xsubj' in cand:
    #             nsubjXsubj.append(cand)
    #         else:
    #             nsubj.append(cand)
    #
    # if len(nsubj) > 0:
    #     return nsubj
    # if len(nsubjXsubj) > 0:
    #     return nsubjXsubj
    #
    # min_distance = 99
    # best_candidate = None
    # for cand in candidates:
    #     print('Considering {}'.format(cand))
    #     parts = cand.split(':')
    #     dependency_distance_from_target = int(parts[0])
    #     up_down = parts[1]
    #     dependency_relation_name = ':'.join(parts[2:-2])
    #     dependency_token_index = int(parts[-2])
    #     dependency_token_text = parts[-1]
    #     if abs(dependency_token_index - target_token.index_in_sentence) < min_distance:
    #         min_distance = abs(dependency_token_index - target_token.index_in_sentence)
    #         best_candidate = cand
    # #print('Returning {}'.format(best_candidate))
    # return [best_candidate]


# def find_shortest_dep_paths_between_tokens(source_token, target_token, sentence):
#     """
#     :type source_token: nlplingo.text.text_span.Token
#     :type target_token: nlplingo.text.text_span.Token
#     :type sentence: nlplingo.text.text_span.Sentence
#     Returns: list[str]
#     """
#     print('target token index = {}'.format(target_token.index_in_sentence))
#     max_length = 5
#     all_paths = []
#     """:type: list[list[nlplingo.text.text_span.DependencyRelation]]"""
#     current_path = []
#     find_dep_paths_from_token(source_token, max_length, current_path, all_paths, sentence.tokens)
#     print('len(all_paths)={}'.format(len(all_paths)))
#     for path in all_paths:
#         path_string = ' '.join('{}:{},{}'.format(r.dep_name, sentence.tokens[r.parent_token_index].text, r.parent_token_index) for r in path)
#         print(' - {}'.format(path_string))
#
#     target_paths = []
#     """:type: list[list[nlplingo.text.text_span.DependencyRelation]]"""
#     for path in all_paths:
#         for i, r in enumerate(path):
#             if r.parent_token_index == target_token.index_in_sentence:
#                 target_paths.append(path[0:i+1])
#                 break
#     print('len(target_paths)={}'.format(len(target_paths)))
#
#     min_path_length = 99
#     min_paths = set()
#     for path in target_paths:
#         path_string = ' '.join('{}:{}'.format(r.dep_name, sentence.tokens[r.parent_token_index].text) for r in path)
#         path_len = len(path)
#         if path_len < min_path_length:
#             min_path_length = path_len
#             min_paths.clear()
#             min_paths.add(path_string)
#         elif path_len == min_path_length:
#             min_paths.add(path_string)
#
#     return min_paths


def add_corenlp_annotations(doc, filepath):
    """Reads Stanford corenlp annotations from filename, and add to doc

    :type filepath: str
    :type doc: nlplingo.text.text_theory.Document
    """
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    for sentence_json in json_data['sentences']:
        (index, tokens) = _read_sentence(sentence_json)
        sent_start = tokens[0].start_char_offset()
        sent_end = tokens[-1].end_char_offset()
        sent_text = doc.text[sent_start:sent_end]

        s = Sentence(doc.docid, IntPair(sent_start, sent_end), sent_text, tokens, index)
        doc.add_sentence(s)

    return doc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file')
    parser.add_argument('--corenlp_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    with codecs.open(args.text_file, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    doc = Document(os.path.basename(args.text_file), text=doc_text)
    doc = add_corenlp_annotations(doc, args.corenlp_file)

    sentence_strings = []
    for sentence in doc.sentences:
        sentence_string = ' '.join(token.text for token in sentence.tokens)
        sentence_strings.append(sentence_string)

    with codecs.open(args.output_file, 'w', encoding='utf-8') as o:
        for s in sentence_strings:
            o.write('{}\n'.format(s))
