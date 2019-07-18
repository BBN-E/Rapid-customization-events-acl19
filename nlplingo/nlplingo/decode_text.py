from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import argparse
import codecs
import spacy
import json

import pickle

from collections import defaultdict

import numpy as np

from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_span import EventArgument
from nlplingo.text.text_theory import Document
from nlplingo.annotation.idt import process_idt_file
from nlplingo.annotation.enote import process_enote_file
from nlplingo.embeddings.word_embeddings import WordEmbedding
from nlplingo.model.extractor import Extractor
from nlplingo.event.event_trigger import EventTriggerGenerator
from nlplingo.event.event_argument import EventArgumentGenerator
from nlplingo.event.event_domain import EventDomain
from nlplingo.event.event_domain import CyberDomain
from nlplingo.event.event_domain import AceDomain
from nlplingo.event.event_domain import PrecursorDomain
import nlplingo.common.utils as text_utils
from nlplingo.common.parameters import Parameters
from nlplingo.common.utils import IntPair
from nlplingo.common.io_utils import ComplexEncoder

from nlplingo.common.scoring import evaluate_f1

from nlplingo.event.train_test import generate_trigger_data_feature
from nlplingo.event.train_test import generate_argument_data_feature
from nlplingo.event.train_test import load_trigger_modelfile
from nlplingo.event.train_test import load_argument_model
from nlplingo.event.train_test import get_predicted_positive_triggers

from nlplingo.ner.ner_feature import NerFeature
from nlplingo.ner.decoder import Decoder
from nlplingo.ner.decoder import decode_sentence

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    parser.add_argument('--input_filelist')     # a list of text files to decode on
    parser.add_argument('--output_dir')
    parser.add_argument('--params_json')

    args = parser.parse_args()

    params = json.load(open(args.params, 'r'))
    print(json.dumps(params, sort_keys=True, indent=4))

    # load word embeddings
    word_embeddings = WordEmbedding(
        params['embeddings']['embedding_file'],
        int(params['embeddings']['vocab_size']),
        int(params['embeddings']['vector_size']),
        params['embeddings']['none_token'],
        params['embeddings']['missing_token']
    )

    ner_fea = NerFeature(params)
    ner_decoder = Decoder(params)

    spacy_en = spacy.load('en')

    extractors = []
    """:type: list[nlplingo.model.extractor.Extractor]"""
    for extractor_params in params['extractors']:
        extractors.append(Extractor(params, extractor_params, word_embeddings))

    # initialize a particular event domain, which stores info on the event types and event roles
    # event_domain = None
    # if params.get_string('domain') == 'cyber':
    #     event_domain = CyberDomain()
    # elif params.get_string('domain') == 'ace':
    #     event_domain = AceDomain()
    # elif params.get_string('domain') == 'precursor':
    #     event_domain = PrecursorDomain()
    # elif params.get_string('domain') == 'cyberattack':
    #     event_domain = EventDomain.read_domain_ontology_file(params.get_string('domain_ontology'), 'ui')
    #
    # trigger_generator = EventTriggerGenerator(event_domain, params)
    print('==== Loading Trigger model ====')
    trigger_model = load_trigger_modelfile(extractors[0].model_file)

#    if params.has_key('argument_model_dir'):
#        arg_generator = EventArgumentGenerator(event_domain, params)
#        print('==== Loading Argument model ====')
#        argument_model = load_argument_model(params.get_string('argument_model_dir'))
#    else:
#       arg_generator = None
#        argument_model = None

    argument_extractors = []
    if args.params_json is not None:
        with open(args.params_json, 'r') as f:
            params_json = json.load(f)
        for par in params_json['extractors']:
            extractor = Extractor(par['model_type'])
            extractor.domain = EventDomain.read_domain_ontology_file(par['domain_ontology'], par['domain'])
            extractor.model_file = par['model_file']
            extractor.generator = EventArgumentGenerator(extractor.domain, params)
            if extractor.model_type.startswith('event-argument'):
                argument_extractors.append(extractor)

    # read in the list of input files to decode on
    input_filepaths = []
    with codecs.open(args.input_filelist, 'r', encoding='utf-8') as f:
        for line in f:
            input_filepaths.append(line.strip())

    for i, input_filepath in enumerate(input_filepaths):
        print('decoding {} of {} files'.format(i, len(input_filepaths)))

        docid = os.path.basename(input_filepath)
        with codecs.open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        spacy_doc = spacy_en(content)
        ner_predictions = []
        for sent in spacy_doc.sents:
            ner_predictions.extend(
                decode_sentence(
                    ner_fea,
                    ner_decoder,
                    content,
                    sent,
                    offset=0,
                    content_type=params['content_type']
                )
            )


        # create a document based on text content, add NER predictions as EntityMentions, then apply Spacy to
        # perform sentence segmentation and tokenization, and use Spacy tokens to back the EntityMentions
        doc = Document(docid, content)
        for i, p in enumerate(ner_predictions):
            id = 'em-{}'.format(i)
            doc.add_entity_mention(EntityMention(id, IntPair(p['start'], p['end']), p['text'], p['label']))
        doc.annotate_sentences(word_embeddings, spacy_en)

        (trigger_examples, trigger_data, trigger_data_list, trigger_label) = generate_trigger_data_feature(
            extractors[0].generator,
            [doc],
            extractors[0].model_type,
            extractors[0].model_flags['use_bio_index']
        )
        predicted_positive_triggers = []
        if len(trigger_examples) > 0:
            trigger_predictions = trigger_model.predict(trigger_data_list)
            #print(trigger_predictions)
            predicted_positive_triggers = get_predicted_positive_triggers(trigger_predictions, trigger_examples,
                                                                  extractors[0].domain.get_event_type_index('None'),
                                                                  extractors[0].domain)
        print('len(predicted_positive_triggers)={}'.format(len(predicted_positive_triggers)))

        predicted_events = defaultdict(list)  # to collate by anchor
        if len(argument_extractors) > 0 and argument_extractors[0].generator is not None and argument_extractors[0].domain is not None:
            if len(predicted_positive_triggers) > 0:
                # generate arguments with predicted triggers
                (arg_examples_pt, arg_data_pt, arg_data_list_pt, arg_label_pt) = \
                    generate_argument_data_feature(argument_extractors[0].generator, [doc], params, predicted_triggers=predicted_positive_triggers)

                # decode arguments with predicted triggers
                pred_arg_max = []
                if len(arg_examples_pt) > 0:
                    argument_predictions_pt = argument_extractors[0].model.predict(arg_data_list_pt)
                    pred_arg_max = np.argmax(argument_predictions_pt, axis=1)

                for i, predicted_label in enumerate(pred_arg_max):
                    if predicted_label != extractors[0].domain.get_event_role_index('None'):
                        eg = arg_examples_pt[i]
                        """:type: nlplingo.event.event_argument.EventArgumentExample"""
                        predicted_role = extractors[0].domain.get_event_role_from_index(predicted_label)
                        #print('{} || {} || {}'.format(predicted_role, eg.anchor.to_string(), eg.argument.to_string()))
                        predicted_events[eg.anchor].append(EventArgument('dummy', eg.argument, predicted_role))
        else:
            for docid in predicted_positive_triggers:
                for trigger_example in predicted_positive_triggers[docid]:
                    predicted_events[trigger_example.anchor] = []

        events_lines = []
        if len(predicted_events) > 0:
            for sent in doc.sentences:
                """:type: nlplingo.text.text_span.Sentence"""
                sent_start = sent.start_char_offset()
                sent_end = sent.end_char_offset()

                for anchor in predicted_events.keys():
                    """:type: nlplingo.text.text_span.Anchor"""
                    if sent_start<=anchor.start_char_offset() and anchor.end_char_offset()<=sent_end:
                        e = dict()
                        e['anchor_start'] = anchor.start_char_offset()
                        e['anchor_end'] = anchor.end_char_offset()
                        e['anchor_text'] = anchor.text
                        e['event_type'] = anchor.label
                        e['sentence'] = sent.text

                        arguments = []
                        for arg in predicted_events[anchor]:
                            """:type: nlplingo.text.text_span.EventArgument"""
                            d = dict()
                            d['arg_start'] = arg.start_char_offset()
                            d['arg_end'] = arg.end_char_offset()
                            d['arg_text'] = arg.text
                            d['event_role'] = arg.label
                            arguments.append(d)
                        e['arguments'] = arguments
                        events_lines.append(e)

        with codecs.open('{}/{}'.format(args.output_dir, docid), 'w', encoding='utf-8') as o:
            o.write(json.dumps(events_lines, indent=4, cls=ComplexEncoder, ensure_ascii=False))

