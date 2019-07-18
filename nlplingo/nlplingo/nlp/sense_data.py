import sys
import os
import codecs
from collections import defaultdict
import re
import json
import argparse

from nlplingo.common.parameters import Parameters
from nlplingo.common.utils import IntPair
from nlplingo.common.utils import write_list_to_file

from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import Event

from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import Token
from nlplingo.text.text_span import Anchor

from nlplingo.annotation.spannotator import SpannotatorAnnotation


class SenseData(object):

    def __init__(self, params):
        """
        :type params: nlplingo.common.parameters.Parameters

        Format of sense.event_senseid_file:
        [
            {
                ...
                "sense_ids": [
                    "birth%2:29:00::",
                    "birth%1:11:00::",
                    "birth%1:22:00::"
                ],
                "event_type": "Life.Be-Born"
            },
            ...
        ]

        Format of sense.annotation_file:
        [
            {
                "COLLECTION": "semcor",
                "DOCUMENT_ID": "br-k09",
                "GUID": "[semcor][br-k09][0]",
                "TOKENIZED_TEXT": "It was the first time any of us had laughed since the morning began .",
                "EXAMPLES": [
                    {
                        "lemma": "be",
                        "lexsn": "2:42:03::",
                        "wnsn": "1",
                        "start_token_index": 1,
                        "end_token_index": 1,
                        "pos_tag": "VB",
                        "duplicate": false
                    },
                    {
                        "lemma": "first",
                        "lexsn": "3:00:00::",
                        "wnsn": "1",
                        "start_token_index": 3,
                        "end_token_index": 3,
                        "pos_tag": "JJ",
                        "duplicate": false
                    },
                    {
                        "lemma": "time",
                        "lexsn": "1:11:00::",
                        "wnsn": "1",
                        "start_token_index": 4,
                        "end_token_index": 4,
                        "pos_tag": "NN",
                        "duplicate": false
                    },
                    ...
                ]
            },
            ...
        ]
        """
        self.senseid_to_et = self._read_event_senseid_file(params.get_string('sense.event_senseid_file'))
        self.documents = self._read_sense_annotation_file(params.get_string('sense.annotation_file'))


    def _read_sense_annotation_file(self, filepath):

        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        documents = dict()  # docid -> Document
        for sentence_data in datas:
            docid = sentence_data['DOCUMENT_ID']

            if docid not in documents.keys():
                documents[docid] = Document(docid)
            doc = documents[docid]

            sentence = self._construct_sentence(sentence_data, doc)
            self._add_sense_examples_as_triggers(sentence_data['EXAMPLES'], sentence)
            if len(sentence.events) > 0:
                doc.add_sentence(sentence)

        return documents.values()


    def _add_sense_examples_as_triggers(self, examples_data, sentence):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type doc: nlplingo.text.text_theory.Document
        """
        for example_data in examples_data:
            start_token_index = example_data['start_token_index']
            end_token_index = example_data['end_token_index']
            pos_tag = example_data['pos_tag']
            lemma = example_data['lemma']
            sense_id = lemma + '%' + example_data['lexsn']

            if start_token_index != end_token_index:
                continue
            assert start_token_index < len(sentence.tokens)

            token = sentence.tokens[start_token_index]
            token.lemma = lemma
            token.pos_tag = pos_tag

            if sense_id in self.senseid_to_et:
                event_type = self.senseid_to_et[sense_id]
                anchor = Anchor('dummy', IntPair(token.start_char_offset(), token.end_char_offset()), token.text, event_type)
                event = Event('dummy', event_type)
                event.add_anchor(anchor)
                sentence.add_event(event)


    def _construct_sentence(self, sentence_data, doc):
        text = sentence_data['TOKENIZED_TEXT'].strip()
        text = re.sub('\s+', ' ', text).strip()

        offset = doc.text_length()
        if len(doc.sentences) > 0:
            offset += 1
        sentence_length_thus_far = 0
        tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""
        for i, token_string in enumerate(text.split(' ')):
            start = offset + sentence_length_thus_far   # +1 for newline
            end = start + len(token_string)
            token = Token(IntPair(start, end), i, token_string, lemma=None, pos_tag=None)
            sentence_length_thus_far += len(token_string) + 1  # +1 for space
            tokens.append(token)
        return Sentence(doc.docid, IntPair(tokens[0].start_char_offset(), tokens[-1].end_char_offset()), text, tokens, len(doc.sentences))


    def _read_event_senseid_file(self, filepath):
        ret = dict()
        id_to_et_dict = defaultdict(set)  # sense_id -> set(event_type)

        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        for data in datas:
            et = data['event_type']
            for sense_id in data['sense_ids']:
                id_to_et_dict[sense_id].add(et)

        # each sense id should only be mapped to 1 single event type. Only keep unambiguous ones
        for sense_id in id_to_et_dict.keys():
            if len(id_to_et_dict[sense_id]) == 1:
                ret[sense_id] = list(id_to_et_dict[sense_id])[0]
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    params = Parameters(args.params)
    params.print_params()

    sense_data = SenseData(params)

    path_lines = []
    for doc in sense_data.documents:
        span_lines = SpannotatorAnnotation.print_event_annotations(doc)

        span_file = args.output_dir + '/' + doc.docid + '.span'
        text_file = args.output_dir + '/' + doc.docid + '.txt'
        path_lines.append('SPAN:{} TEXT:{}'.format(span_file, text_file))

        write_list_to_file(span_lines, span_file)
        write_list_to_file([sent.text for sent in doc.sentences], text_file)

    write_list_to_file(path_lines, args.output_dir + '/span_text.list')

