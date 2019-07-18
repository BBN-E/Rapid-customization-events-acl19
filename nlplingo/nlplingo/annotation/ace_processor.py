import os
import sys
import codecs
from collections import defaultdict
import argparse
import re
import json
import glob

from nlplingo.common.io_utils import ComplexEncoder
from nlplingo.common.io_utils import read_file_to_list
from nlplingo.common.io_utils import write_list_to_file
from nlplingo.common.parameters import Parameters

from nlplingo.text.text_theory import Document

from nlplingo.annotation.ace import AceAnnotation
from nlplingo.annotation.stanford_corenlp import add_corenlp_annotations
from nlplingo.annotation.srl_column import add_srl_annotations



class TextSpan(object):
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end


def afp_line_breaks(lines):
    for i in range(len(lines)):
        if '(AFP) ' in lines[i]:
            lines[i] = lines[i].replace('(AFP) ', '(AFP)\n')

def apw_line_breaks(lines):
    for i in range(len(lines)):
        if '(AP) ' in lines[i]:
            lines[i] = lines[i].replace('(AP) ', '(AP)\n')

def xin_line_breaks(lines):
    for i in range(len(lines)):
        if '(Xinhua) ' in lines[i]:
            lines[i] = lines[i].replace('(Xinhua) ', '(Xinhua)\n')

def apply_corpus_line_breaks(lines, filename):
    if filename.startswith('AFP_'):
        afp_line_breaks(lines)
    elif filename.startswith('APW_'):
        apw_line_breaks(lines)
    elif filename.startswith('XIN_'):
        xin_line_breaks(lines)

def read_sentence_tokens(filepath):
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    sentences = []
    for sentence in datas['sentences']:
        tokens = []
        for token in sentence['tokens']:
            text = token['originalText']
            start = token['characterOffsetBegin']
            end = token['characterOffsetEnd']
            tokens.append(TextSpan(text, start, end))
        sentence_text = ' '.join(token.text for token in tokens)
        if len(sentence_text.strip()) > 0:
            sentences.append(TextSpan(sentence_text, tokens[0].start, tokens[-1].end))
    return sentences

def write_filelist_to_json(params, output_file):
    datas = []
    for text_filepath in glob.glob(params.get_string('text.dir') + '/*'):
        filename = os.path.basename(text_filepath)
        assert os.path.isfile(text_filepath)
        offset_filepath = os.path.join(params.get_string('offset.dir'), filename + '.' + params.get_string('offset.ext'))
        assert os.path.isfile(offset_filepath)
        corenlp_filepath = os.path.join(params.get_string('corenlp.dir'), filename + '.' + params.get_string('corenlp.ext'))
        assert os.path.isfile(corenlp_filepath)
        srl_filepath = os.path.join(params.get_string('srl.dir'), filename + '.' + params.get_string('srl.ext'))
        assert os.path.isfile(srl_filepath)

        d = dict()
        d['docid'] = filename
        d['text_file'] = text_filepath
        d['offset_file'] = offset_filepath
        d['corenlp_file'] = corenlp_filepath
        d['srl_file'] = srl_filepath
        datas.append(d)

    with codecs.open(output_file, 'w', encoding='utf-8') as o:
        o.write(json.dumps(datas, sort_keys=True, indent=4, cls=ComplexEncoder, ensure_ascii=False))
        o.close()

def combine_annotations(filelist_json, output_dir):
    with codecs.open(filelist_json, 'r', encoding='utf-8') as f:
        filelist_datas = json.load(f)

    for filelist_data in filelist_datas:
        docid = filelist_data['docid']
        text_file = filelist_data['text_file']
        offset_file = filelist_data['offset_file']
        corenlp_file = filelist_data['corenlp_file']
        srl_file = filelist_data['srl_file']

        with codecs.open(text_file, 'r', encoding='utf-8') as f:
            doc_text = f.read()

        doc = Document(docid, text=doc_text)
        add_corenlp_annotations(doc, corenlp_file)
        add_srl_annotations(doc, srl_file, offset_file)

        doc_d = dict()
        doc_d['docid'] = doc.docid
        doc_d['text'] = doc.text
        sentence_dicts = []
        for sentence in doc.sentences:
            sentence_d = dict()
            sentence_d['index'] = sentence.index
            sentence_d['text'] = sentence.text
            sentence_d['start'] = sentence.start_char_offset()
            sentence_d['end'] = sentence.end_char_offset()

            token_dicts = []
            for token in sentence.tokens:
                token_d = dict()
                token_d['index'] = token.index_in_sentence
                token_d['text'] = token.text
                token_d['start'] = token.start_char_offset()
                token_d['end'] = token.end_char_offset()
                token_d['lemma'] = token.lemma
                token_d['pos_tag'] = token.pos_tag

                dep_dicts = []
                for r in token.dep_relations:
                    dep_d = dict()
                    dep_d['dep_name'] = r.dep_name
                    dep_d['dep_token_index'] = r.connecting_token_index
                    dep_d['dep_token_text'] = sentence.tokens[r.connecting_token_index].text
                    dep_d['dep_direction'] = r.dep_direction
                    dep_dicts.append(dep_d)
                token_d['dep_relations'] = dep_dicts

                srl_dict = dict()
                if token.srl is not None:
                    srl = token.srl
                    srl_dict['predicate'] = srl.predicate_label
                    srl_role_dicts = []
                    for role in srl.roles:
                        for index in srl.roles[role]:
                            d = dict()
                            d['srl_role'] = role
                            d['srl_token_index'] = index
                            d['srl_token_text'] = sentence.tokens[index].text
                            srl_role_dicts.append(d)
                    srl_dict['roles'] = srl_role_dicts
                token_d['srl'] = srl_dict

                token_dicts.append(token_d)
            sentence_d['tokens'] = token_dicts
            sentence_dicts.append(sentence_d)
        doc_d['sentences'] = sentence_dicts

        with codecs.open(os.path.join(output_dir, docid), 'w', encoding='utf-8') as o:
            o.write(json.dumps(doc_d, sort_keys=True, indent=4, cls=ComplexEncoder, ensure_ascii=False))
            o.close()

def replace_infix_newline(line):
    start = 0
    while start < len(line) and (line[start] == ' ' or line[start] == '\n'):
        start += 1

    end = len(line)-1
    while end >= 0 and (line[end] == ' ' or line[end] == '\n'):
        end -= 1

    if start < end:
        return line[0:start] + line[start:end].replace('\n', ' ') + line[end:]
    else:
        return line

def expand_amp(line):
    return line.replace('&', '  &  ') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    if args.mode == 'strip_xml':
        input_filelist = params.get_string('input_filelist')
        output_dir = params.get_string('output_dir')
        for filepath in read_file_to_list(input_filelist):
            print(filepath)
            filename = os.path.basename(filepath)
            filename = re.search(r'^(.*)\.(.*?)$', filename).group(1)

            text_list = AceAnnotation.process_ace_textfile_to_list(filepath)
            for i in range(len(text_list)):
                text_list[i] = replace_infix_newline(text_list[i])
                text_list[i] = expand_amp(text_list[i])

            apply_corpus_line_breaks(text_list, filename)
            with codecs.open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as o:
                o.write("".join(text_list))

    elif args.mode == 'write_sentence_tokens':
        input_filelist = params.get_string('input_filelist')
        output_dir = params.get_string('output_dir')
        for filepath in read_file_to_list(input_filelist):
            filename = os.path.basename(filepath)
            filename = re.search(r'^(.*)\.(.*?)$', filename).group(1)
            sentences = read_sentence_tokens(filepath)

            write_list_to_file([s.text for s in sentences], os.path.join(output_dir, filename + '.txt'))
            write_list_to_file(['{} {}'.format(str(s.start), str(s.end)) for s in sentences], os.path.join(output_dir, filename + '.offset'))

    elif args.mode == 'filelist_json':  # produce a json file listing all the files
        output_file = params.get_string('output_file')
        write_filelist_to_json(params, output_file)

    elif args.mode == 'combine_annotations':
        filelist_json = params.get_string('filelist_json')
        output_dir = params.get_string('output_dir')
        combine_annotations(filelist_json, output_dir)

