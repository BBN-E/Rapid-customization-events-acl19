from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from future.builtins import zip

import math
import os
import xml.etree.ElementTree as etree
from collections import defaultdict
from operator import itemgetter
import re
import numpy as np
import csv

import spacy

import config
from io import open

verbose = 1

# ACE 2005 event types
EVENT = [u'Conflict', u'Life', u'Movement', u'Justice', u'Personnel', u'Contact',
         u'Transaction', u'Business']

# ACE 2005 Event Subtypes
EVENT_SUBTYPE = [
    u'Be-Born', u'Die', u'Marry', u'Divorce', u'Injure', u'Transfer-Ownership',
    u'Transfer-Money', u'Transport', u'Start-Org', u'End-Org', u'Declare-Bankruptcy',
    u'Merge-Org', u'Attack', u'Demonstrate', u'Meet', u'Phone-Write', u'Start-Position',
    u'End-Position', u'Nominate', u'Elect', u'Arrest-Jail', u'Release-Parole',
    u'Charge-Indict', u'Trial-Hearing', u'Sue', u'Convict', u'Sentence', u'Fine',
    u'Execute', u'Extradite', u'Acquit', u'Pardon', u'Appeal']

# ACE 2005 role for events
ROLES = [u'Person', u'Place', u'Buyer', u'Seller', u'Beneficiary', u'Price',
         u'Artifact', u'Origin', u'Destination', u'Giver', u'Recipient', u'Money',
         u'Org', u'Agent', u'Victim', u'Instrument', u'Entity', u'Attacker', u'Target',
         u'Defendant', u'Adjudicator', u'Prosecutor', u'Plaintiff', u'Crime',
         u'Position', u'Sentence', u'Vehicle', u'Time-Within', u'Time-Starting',
         u'Time-Ending', u'Time-Before', u'Time-After', u'Time-Holds',
         u'Time-At-Beginning', u'Time-At-End']

# ACE 2005 roles for each event subtype
EVENT_SUBTYPE_ROLES = {
    u'Acquit': set([u'Defendant', u'Time-Within', u'Adjudicator', u'Crime']),
    u'Appeal': set([u'Adjudicator',
               u'Crime',
               u'Place',
               u'Plaintiff',
               u'Time-Holds',
               u'Time-Within']),
    u'Arrest-Jail': set([u'Agent',
                    u'Crime',
                    u'Person',
                    u'Place',
                    u'Time-At-Beginning',
                    u'Time-Before',
                    u'Time-Ending',
                    u'Time-Holds',
                    u'Time-Starting',
                    u'Time-Within']),
    u'Attack': set([u'Agent',
               u'Attacker',
               u'Instrument',
               u'Place',
               u'Target',
               u'Time-After',
               u'Time-At-Beginning',
               u'Time-At-End',
               u'Time-Before',
               u'Time-Ending',
               u'Time-Holds',
               u'Time-Starting',
               u'Time-Within',
               u'Victim']),
    u'Be-Born': set([u'Time-Within', u'Place', u'Time-Holds', u'Person']),
    u'Charge-Indict': set([u'Adjudicator',
                      u'Crime',
                      u'Defendant',
                      u'Place',
                      u'Prosecutor',
                      u'Time-Before',
                      u'Time-Ending',
                      u'Time-Within']),
    u'Convict': set([u'Adjudicator',
                u'Crime',
                u'Defendant',
                u'Place',
                u'Time-At-Beginning',
                u'Time-Within']),
    u'Declare-Bankruptcy': set([u'Org',
                           u'Place',
                           u'Time-After',
                           u'Time-At-Beginning',
                           u'Time-Within']),
    u'Demonstrate': set([u'Entity',
                    u'Place',
                    u'Time-At-End',
                    u'Time-Starting',
                    u'Time-Within']),
    u'Die': set([u'Agent',
            u'Instrument',
            u'Person',
            u'Place',
            u'Time-After',
            u'Time-At-Beginning',
            u'Time-Before',
            u'Time-Ending',
            u'Time-Holds',
            u'Time-Starting',
            u'Time-Within',
            u'Victim']),
    u'Divorce': set([u'Place', u'Time-Within', u'Person']),
    u'Elect': set([u'Entity',
              u'Person',
              u'Place',
              u'Position',
              u'Time-At-Beginning',
              u'Time-Before',
              u'Time-Holds',
              u'Time-Starting',
              u'Time-Within']),
    u'End-Org': set([u'Org',
                u'Place',
                u'Time-After',
                u'Time-At-Beginning',
                u'Time-Holds',
                u'Time-Within']),
    u'End-Position': set([u'Entity',
                     u'Person',
                     u'Place',
                     u'Position',
                     u'Time-After',
                     u'Time-At-End',
                     u'Time-Before',
                     u'Time-Ending',
                     u'Time-Holds',
                     u'Time-Starting',
                     u'Time-Within']),
    u'Execute': set([u'Agent',
                u'Crime',
                u'Person',
                u'Place',
                u'Time-After',
                u'Time-At-Beginning',
                u'Time-Within']),
    u'Extradite': set([u'Destination', u'Time-Within', u'Origin', u'Agent', u'Person']),
    u'Fine': set([u'Time-Within', u'Adjudicator', u'Place', u'Money', u'Crime', u'Entity']),
    u'Injure': set([u'Place', u'Time-Within', u'Victim', u'Agent', u'Instrument']),
    u'Marry': set([u'Place', u'Time-Within', u'Time-Before', u'Time-Holds', u'Person']),
    u'Meet': set([u'Entity',
             u'Place',
             u'Time-After',
             u'Time-At-Beginning',
             u'Time-Ending',
             u'Time-Holds',
             u'Time-Starting',
             u'Time-Within']),
    u'Merge-Org': set([u'Org', u'Time-Ending']),
    u'Nominate': set([u'Agent', u'Time-Within', u'Position', u'Person']),
    u'Pardon': set([u'Defendant', u'Place', u'Time-At-End', u'Adjudicator']),
    u'Phone-Write': set([u'Entity',
                    u'Place',
                    u'Time-After',
                    u'Time-Before',
                    u'Time-Holds',
                    u'Time-Starting',
                    u'Time-Within']),
    u'Release-Parole': set([u'Crime',
                       u'Entity',
                       u'Person',
                       u'Place',
                       u'Time-After',
                       u'Time-Within']),
    u'Sentence': set([u'Adjudicator',
                 u'Crime',
                 u'Defendant',
                 u'Place',
                 u'Sentence',
                 u'Time-At-End',
                 u'Time-Starting',
                 u'Time-Within']),
    u'Start-Org': set([u'Agent',
                  u'Org',
                  u'Place',
                  u'Time-After',
                  u'Time-Before',
                  u'Time-Starting',
                  u'Time-Within']),
    u'Start-Position': set([u'Entity',
                       u'Person',
                       u'Place',
                       u'Position',
                       u'Time-After',
                       u'Time-At-Beginning',
                       u'Time-Before',
                       u'Time-Holds',
                       u'Time-Starting',
                       u'Time-Within']),
    u'Sue': set([u'Adjudicator',
            u'Crime',
            u'Defendant',
            u'Place',
            u'Plaintiff',
            u'Time-Holds',
            u'Time-Within']),
    u'Transfer-Money': set([u'Beneficiary',
                       u'Giver',
                       u'Money',
                       u'Place',
                       u'Recipient',
                       u'Time-After',
                       u'Time-Before',
                       u'Time-Holds',
                       u'Time-Starting',
                       u'Time-Within']),
    u'Transfer-Ownership': set([u'Artifact',
                           u'Beneficiary',
                           u'Buyer',
                           u'Place',
                           u'Price',
                           u'Seller',
                           u'Time-At-Beginning',
                           u'Time-Before',
                           u'Time-Ending',
                           u'Time-Within']),
    u'Transport': set([u'Agent',
                  u'Artifact',
                  u'Destination',
                  u'Origin',
                  u'Place',
                  u'Time-After',
                  u'Time-At-Beginning',
                  u'Time-At-End',
                  u'Time-Before',
                  u'Time-Ending',
                  u'Time-Holds',
                  u'Time-Starting',
                  u'Time-Within',
                  u'Vehicle',
                  u'Victim']),
    u'Trial-Hearing': set([u'Adjudicator',
                      u'Crime',
                      u'Defendant',
                      u'Place',
                      u'Prosecutor',
                      u'Time-At-End',
                      u'Time-Holds',
                      u'Time-Starting',
                      u'Time-Within'])}

ENTITY_TYPE = [u'FAC', u'PER', u'LOC', u'GPE', u'ORG', u'WEA', u'VEH']
VALUE_TYPE = [u'Sentence', u'Job-Title', u'Crime', u'Contact-Info', u'Numeric']
TIME_TYPE = [u'Time']

ENTITY_VALUE_TIME = ENTITY_TYPE + VALUE_TYPE + TIME_TYPE
ENTITY_VALUE_TIME_SIZE = len(ENTITY_VALUE_TIME)

ENTITY_VALUE_LOOKUP = dict()
for i, x in enumerate(ENTITY_VALUE_TIME):
    ENTITY_VALUE_LOOKUP[x] = i

BIO_ENTITY_TYPE = [u'O'] + [u'B-'+x for x in ENTITY_VALUE_TIME] + [u'I-'+x for x in ENTITY_VALUE_TIME]

def get_bio_index(type_name, is_begin):
    u"""Retrun integer index corresponding to  BIO entity annotation"""
    if is_begin:
        return ENTITY_VALUE_LOOKUP[type_name] + 1
    else:
        return ENTITY_VALUE_LOOKUP[type_name] + ENTITY_VALUE_TIME_SIZE + 1

def extract_text(node, replace_newline_with_space=False):
    return extract_text2(node, [], replace_newline_with_space)

def extract_text2(node, all_text, replace_newline_with_space):
    u""" Return list of text from XML element subtree.
    Python 2 version"""
    tag = node.tag
    if not isinstance(tag, str) and tag is not None:
        return
    text = node.text
    if text:
        if replace_newline_with_space:
            text = text.replace('\n', ' ')
        all_text.append(unicode(text))
    for e in node:
        extract_text2(e, all_text, replace_newline_with_space)
        text = e.tail
        if text:
            if replace_newline_with_space:
                text = text.replace('\n', ' ')
            all_text.append(unicode(text))
    return all_text

def extract_text3(node, replace_newline_with_space):
    u""" Return list of text from XML element subtree.
    Python 3 version"""
    all = [];
    for text in node.itertext():
        if replace_newline_with_space:
            text = text.replace('\n', ' ')
        all.append(text)
    return all

def find_sentence_index_by_string(search_string, start_idx, end_idx, all_text, sent_text_idx):
    u""" Find the sentence containing 'search_string'.
    Use 'start_idx' and 'end_idx' of 'all_text' as hints to the location of the sentence.
    'sent_text_idx' is a list of pairs indicating the beginning and end of sentences.
    """    
    text_string = all_text[start_idx:end_idx]
    if not text_string == search_string:
        best_match = (len(all_text), None)
        start = 0
        while True:
            match_pos = all_text.find(search_string, start)
            if match_pos < 0:
                break
            dist = abs(start_idx - match_pos)
            if dist < best_match[0]:
                best_match = (dist, match_pos)
            start = match_pos + 1
        # for match in re.finditer(search_string, all_text):
        #     dist = abs(start_idx - match.start())
        #     if dist < best_match[0]:
        #         best_match = (dist, match)
        if best_match[1]:
            if verbose:
                print(u' Search string and indices mismatch: "{0}" != "{1}". ' +
                      u'Found match by shifting {2} chars'.format(
                          search_string, all_text[start_idx:end_idx],
                          start_idx-best_match[1]))
            start_idx = best_match[1]
            end_idx = best_match[1]+len(search_string)
            # if verbose:
            #     print(' Search string and indices mismatch: "{}" != "{}". Found match by shifting {} chars'.format(
            #         search_string, all_text[start_idx:end_idx], start_idx-best_match[1].start()))
            # start_idx = best_match[1].start()
            # end_idx = best_match[1].end()
        else:
            print(u' !! Search string ({0}) not in text.'.format(search_string))
            return -1
    sent_idx = [i for i in xrange(len(sent_text_idx))
               if (sent_text_idx[i][0] <= start_idx and
                   end_idx <= sent_text_idx[i][1])]
    if len(sent_idx) == 0:
        if verbose:
            print(u' !! Search string ({0}) not in sentence.'.format(search_string))
        return -1
    if len(sent_idx) > 1:
        print(u' !! findSentByString: Multiple sentence matches for {0}'.format(search_string))
    return sent_idx[0]

def file_iterator(data_dir, suffix):
    for dir_name, subDirs, files in os.walk(data_dir):
        for afile in [x for x in files if x.endswith(suffix)]:
            yield (dir_name, afile)

def filelist_iterator(data_dir, file_base_list, suffix):
    for base in file_base_list:
        yield (data_dir, base + suffix)
    
    
def read_file_prefixes(filename):
    result = []
    with open(filename, u'r') as f:
        for prefix in f:
            name = prefix.strip()
            result.append(name)
    return result
    
def sentence_stats(data_dir, display=0, nlp=None, num_tokens_long_sent=4):
    sentence_stats_by_iter(file_iterator(data_dir, u'.sgm'), display=display, nlp=nlp,
                        num_tokens_long_sent=num_tokens_long_sent)
    
def sentence_stats_by_iter(sgml_file_iter, display=0, nlp=None, n_threads=8, num_tokens_long_sent=4):
    if nlp is None:
        nlp = spacy.load(u'en')
    sent_count = 0
    long_sent_count = 0
    max_length = 0
    sum_length = 0
    sum_squared_length = 0
    length_list = []
    docs = []
    psum = [0]

    for dir_name, sgml_file in sgml_file_iter:
        if display > 1:
            print(sgml_file)
        tree = etree.parse(os.path.join(dir_name, sgml_file))
        root = tree.getroot()
        text_list = extract_text(root)
        for i, doc in enumerate(nlp.pipe(text_list, batch_size=10,
                                         n_threads=n_threads)):
            docs.append(doc)
            psum.append(psum[-1] + len(text_list[i]))
            for span in doc.sents:
                sent_count += 1
                num_tokens = span.end - span.start
                if num_tokens > max_length:
                    max_length = num_tokens
                if num_tokens >= num_tokens_long_sent:
                    length_list.append(num_tokens)
                    long_sent_count += 1
                    sum_length += num_tokens
                    sum_squared_length += num_tokens * num_tokens
                    if display > 2:
                        print(num_tokens)
                        sent = u''.join(doc[i].string for i in xrange(span.start, span.end)).strip()
                        print(sent)
    print(u'Sentence statistics (ignoring short sentences <{0} tokens):'.format(
        num_tokens_long_sent))
    print(u'Number of sentences: {0}'.format(sent_count))
    print(u'Number of long sentences: {0}'.format(long_sent_count))
    print(u'Max long sentence length: {0}'.format(max_length))
    print(u'Average long sentence length: {0:.2f}'.format(sum_length/long_sent_count))
    std = math.sqrt((sum_squared_length - (sum_length*sum_length)
                     /long_sent_count)/(long_sent_count-1))
    print(u'Std long sentence length: {0:.2f}'.format(std))
    length_list.sort()
    larray = np.asarray(length_list)
    larray = np.floor((1+larray)/10)
    d = defaultdict(int)
    for x in larray:
        d[x] += 1
    print(u'Length distribution')
    for k in sorted(d.keys()):
        print(u' {0:3}: {1:5} {2:6.3f}'.format(int(10*k), d[k], d[k]/len(length_list)))

def entity_stats(apf_xml_file_iter, display=0):
    type_count = defaultdict(lambda: defaultdict(int))
    for dir_name, xmlfile in apf_xml_file_iter:
        if display > 1:
            print(xmlfile)
        tree = etree.parse(os.path.join(dir_name, xmlfile))
        root = tree.getroot()
        entities = root[0].findall(u'entity')
        for entity in entities:
            etype = entity.attrib[u'TYPE']
            esubtype = entity.attrib[u'SUBTYPE']
            sub_dict = type_count[etype]
            sub_dict[esubtype] += 1
    if display > 0:
        for etype in type_count.keys():
            sub_dict = type_count[etype]
            total = sum(sub_dict.values())
            print(u' {0}: {1}'.format(etype, total))
            for key, value in sub_dict.items():
                print(u'   {0}: {1}'.format(key, value))
    return type_count

def value_stats(apf_xml_file_iter, display=0):
    type_count = defaultdict(lambda: defaultdict(int))
    for dir_name, xmlfile in apf_xml_file_iter:
        if display > 1:
            print(xmlfile)
        tree = etree.parse(os.path.join(dir_name, xmlfile))
        root = tree.getroot()
        entities = root[0].findall(u'value')
        for entity in entities:
            etype = entity.attrib[u'TYPE']
            sub_dict = type_count[etype]
            if u'SUBTYPE' in entity.attrib:
                esubtype = entity.attrib[u'SUBTYPE']
            else:
                esubtype = u'None'
            sub_dict[esubtype] += 1
    if display > 0:
        for etype in type_count.keys():
            sub_dict = type_count[etype]
            total = sum(sub_dict.values())
            print(u' {0}: {1}'.format(etype, total))
            for key, value in sub_dict.items():
                print(u'   {0}: {1}'.format(key, value))
    return type_count

    
def event_stats(data_dir, display=0):
    event_stats_by_iter(file_iterator(data_dir, u'apf.xml'), display=display)

def event_stats_by_partition(data_dir, display=0):
    list_dict = read_hengji_partition_lists()
    for name in list_dict.keys():
        print(u'== Parition = ' + name)
        event_stats_by_iter(filelist_iterator(data_dir, list_dict[name], u'.apf.xml'), display=display)
        print()

def event_stats_by_iter(apf_xml_file_iter, display=0):
    u"""Return statistics on events, event type and event subtype"""
    file_count = 0
    event_count = 0
    event_mention_count = 0
    event_mention_argument_count = 0
    anchor_word_count = 0
    type_count = defaultdict(int)
    subtype_count = defaultdict(int)
    dir_count = defaultdict(int)
    anchor_word_dist = [0] * 10
    for dir_name, xmlfile in apf_xml_file_iter:
        if display > 1:
            print(xmlfile)
        tree = etree.parse(os.path.join(dir_name, xmlfile))
        root = tree.getroot()
        events = root[0].findall(u'event')
        for event in events:
            type_count[event.attrib[u'TYPE']] += 1
            subtype_count[event.attrib[u'SUBTYPE']] += 1
            dir_count[dir_name] += 1
            event_mentions = event.findall(u'event_mention')
            for mention in event_mentions:
                event_mention_count += 1
                arguments = mention.findall(u'event_mention_argument')
                event_mention_argument_count += len(arguments)
                anchor = mention.find(u'anchor')
                anchor_words = anchor[0].text.split(u' ')
                num_words = sum([len(x)>0 for x in anchor_words])
                if verbose and num_words > 2: # print multi-work anchors
                    print('multi-word anchor: {}'.format(anchor_words))
                anchor_word_dist[num_words if num_words < 10 else 9] += 1
                anchor_word_count += num_words
        file_count += 1
        event_count += len(events)
    if display > 0:
        print(u'Number of apf.xml files: {0}'.format(file_count))
        print(u'Number of events: {0}'.format(event_count))
        print(u'Number of event mentions: {0}'.format(event_mention_count))
        print(u'Number of event mention arguments: {0}'.format(event_mention_argument_count))
        print(u'Average anchor length: {0:.3f}'.format(anchor_word_count/event_mention_argument_count))
        print(u'Anchor length distribution: ', u','.join([unicode(x) for x in anchor_word_dist]))
        print(u'Types')
        for etype, count in sorted([(etype, count)
                                    for (etype, count) in type_count.items()],
                                   key=itemgetter(1), reverse=True):
            print(u'  {0}: {1:.4f}'.format(etype, count/event_count))
        print(u'Subtypes')
        for etype, count in sorted([(etype, count)
                                    for (etype, count) in subtype_count.items()],
                                   key=itemgetter(1), reverse=True):
            print(u'  {0}: {1:.4f}'.format(etype, count/event_count))
        print(u'Directory:')
        for etype, count in sorted([(etype, count)
                                    for (etype, count) in dir_count.items()],
                                   key=itemgetter(1), reverse=True):
            print(u'  {0}: {1:.4f}'.format(etype, count/event_count))
    return (event_count, type_count, subtype_count)

def ace_eval(prediction, label, doutput, num_skipped_events=None, num_total_events=None, additional_predictions=0):
    """
    ('- label=', array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       ...,
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 1]], dtype=int32))

    label: matrix of size (#instance, #label-types)
    So doing an argmax along 2nd dimension amounts to
    extracting the index of the true/predicted label, for each instance

    ('- label_arg_max=', array([3, 2, 3, ..., 3, 3, 3])

    :param prediction:
    :param label:
    :param doutput: number of label types
    :param num_skipped_events:
    :param num_total_events:
    :param additional_predictions:
    :return:
    """
    print('\nace_utils.py : ace_eval()')
    print('- prediction=', prediction)
    print('- label=', label)
    print('- doutput=', doutput)
    print('- num_skipped_events=', num_skipped_events)
    print('- num_total_events=', num_total_events)
    print('- additional_predictions=', additional_predictions)

    # number of instances in data set
    num_instances = label.shape[0]

    # doutput: number of label types
    none_class_index = doutput -1

    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    # label_arg_max is a vector of size #examples
    # pred_arg_max is a vector of size #examples
    # you can now directly compare these label_arg_max vs pred_arg_max
    # to see how many of their elements match. These are the examples that we predicted correctly.

    print('- none_class_index=', none_class_index)
    print('- label_arg_max=', label_arg_max)
    print('- pred_arg_max=', pred_arg_max)

    # check whether each element in label_arg_max != none_class_index
    # So event_instances is a 1-dim vector of size #instances,
    # where each element is True or False
    event_instances = label_arg_max != none_class_index
    print('- event_instances=', event_instances)

    # sum up the number of True elements to obtain the num# of true events
    num_events = np.sum(event_instances)

    if num_skipped_events:
        if num_total_events:
            assert num_total_events == num_events + num_skipped_events
        else:
            num_total_events = num_events + num_skipped_events
    else:
        if num_total_events:
            num_skipped_events = num_total_events - num_events
        else:
            num_total_events = num_events
            num_skipped_events = 0
    print('- num_skipped_events=', num_skipped_events)

    accuracy = np.sum(pred_arg_max==label_arg_max)/num_instances

    correct_event_predictions = pred_arg_max[event_instances] == label_arg_max[event_instances]
    # since event_instances is a vector of True/False elements,
    # pred_arg_max[event_instances] does the following:
    # ret = []
    # for i, v in enumerate(event_instances):
    #     if v == True:
    #         ret.append(pred_arg_max[i])
    #
    # correct_event_predictions : stores whether each element of
    # pred_arg_max[event_instances] matches each element label_arg_max[event_instances]
    print('- pred_arg_max[event_instances]=', pred_arg_max[event_instances])
    print('- label_arg_max[event_instances]=', label_arg_max[event_instances])
    print('- correct_event_predictions=', correct_event_predictions)

    precision = np.sum(correct_event_predictions) / (np.sum(pred_arg_max != none_class_index)
                                                   + additional_predictions)
    recall = np.sum(correct_event_predictions) / (num_events + num_skipped_events)
    f1 = 2.0*precision*recall/(precision+recall)

    print(u'  number of events = {0}'.format(num_events))
    print(u'  number of events (including skipped) = {0}'.format(num_events + num_skipped_events))
    print(u'  number of event prediction = {0}'.format(np.sum(pred_arg_max != none_class_index)))
    print(u'  number of event prediction (including additional) = {0}'.format(
        np.sum(pred_arg_max != none_class_index) + additional_predictions))
    print(u'  number of correct event prediction = {0}'.format(np.sum(correct_event_predictions)))
    print(u'  classification accuracy  = {0}'.format(accuracy))
    print(u'  classification f1        = {0}'.format(f1))
    print(u'  classification precision = {0}'.format(precision))
    print(u'  classification recall    = {0}'.format(recall))

    # True, if instance is event
    ident_label = label_arg_max < none_class_index
    # True, if predict event
    ident_pred = pred_arg_max < none_class_index
    ident_accuracy = 1.0 * np.sum(ident_pred==ident_label) / num_instances
    num_correct_identifications = np.sum(ident_pred[event_instances]==True)
    denom = np.sum(ident_pred) + additional_predictions
    if denom < 1e-8:
        denom = 1e-8
    ident_precision = 1.0 * num_correct_identifications / denom
    ident_recall = 1.0 * num_correct_identifications / (num_events + num_skipped_events)
    if ident_precision < 1e-8:
        ident_precision = 1e-8
    if ident_recall < 1e-8:
        ident_recall = 1e-8
    ident_f1 = 2.0*ident_precision*ident_recall/(ident_precision+ident_recall)

    print('')
    print(u'  number of correct event identication = {0}'.format(np.sum(num_correct_identifications)))
    print(u'  identification accuracy  = {0}'.format(ident_accuracy))
    print(u'  identification f1        = {0}'.format(ident_f1))
    print(u'  identification precision = {0}'.format(ident_precision))
    print(u'  identification recall    = {0}'.format(ident_recall))
    
    result = {}
    result[u'accuracy'] = accuracy
    result[u'precision'] = precision
    result[u'recall'] = recall
    result[u'f1'] = f1
    result[u'identification-accuracy'] = ident_accuracy
    result[u'identification-precision'] = ident_precision
    result[u'identification-recall'] = ident_recall
    result[u'identification-f1'] = ident_f1
    return result

def ace_eval_cutoff(prediction_prob, label, class_index, cutoff, num_skipped_events=0):
    # number of instances in data set
    num_instances = label.shape[0]

    # none_class_index = doutput -1
    ground_truth = label[:,class_index] == 1

    predicted = prediction_prob[:,class_index] > cutoff

    # instances that are actual events
    event_instances = ground_truth

    accuracy = np.sum(predicted==ground_truth)/num_instances

    correct_event_predictions = predicted[event_instances] == ground_truth[event_instances]
    precision = np.sum(correct_event_predictions) / np.sum(predicted)
    recall = np.sum(correct_event_predictions) / (np.sum(event_instances) + num_skipped_events)

    print(u'  number of events = {0}'.format(np.sum(event_instances)))
    print(u'  number of events (including skipped) = {0}'.format(np.sum(event_instances) + num_skipped_events))
    print(u'  number of event prediction = {0}'.format(np.sum(predicted)))
    print(u'  number of correct event prediction = {0}'.format(np.sum(correct_event_predictions)))
    print(u'  accuracy  = {0}'.format(accuracy))
    print(u'  f1        = {0}'.format(2 * (1 / ((1/recall)+(1/precision)))))
    print(u'  precision = {0}'.format(precision))
    print(u'  recall    = {0}'.format(recall))
    result = {}
    result[u'accuracy'] = accuracy
    result[u'precision'] = precision
    result[u'recall'] = recall
    result[u'f1'] = 2 * (1 / ((1/recall)+(1/precision)))
    return result

def role_stats(apf_xml_filer_iter=file_iterator(config.ace_data_dir, u'apf.xml'), display_detail=1):
    stat = defaultdict(int)
    event_roles = dict()
    for event in EVENT_SUBTYPE:
        event_roles[event] = set()
    for dir_name, xmlfile in apf_xml_filer_iter:
        stat[u'numFiles'] += 1
        if display_detail > 1:
            print(xmlfile)
        role_file_stat(os.path.join(dir_name, xmlfile), stat, event_roles, display_detail+1 if display_detail else 0)
    for k in [u'numFiles', u'num_events', u'numEventMentions', u'numRoles', u'compoundAnchor', u'compoundRole']:
        print(u'{0}: {1}'.format(k, stat[k]))
    if display_detail > 0:
        for event_subtype, rset in event_roles.items():
            print(event_subtype)
            roles = list(rset)
            roles.sort()
            print(u' ' + u','.join(roles))
    return (stat, event_roles)
            
def role_file_stat(filename, stat, event_roles, display_detail):
    tree = etree.parse(filename)
    root = tree.getroot()
    document_node = root[0]
    all_events = document_node.findall(u'event')
    for event in all_events:
        all_arguments = event.findall(u'event_argument')
        for argument in all_arguments:
            event_roles[event.attrib[u'SUBTYPE']].add(argument.attrib[u'ROLE'])
        stat[u'num_events'] += 1
        if display_detail > 2:
            print(u' '*display_detail + u'Event id={0} type={1} subtype={2}'.format(
                event.attrib[u'ID'], event.attrib[u'TYPE'], event.attrib[u'SUBTYPE']))
        role_event_stat(event, document_node, stat, display_detail+1 if display_detail else 0)

def role_event_stat(event, document_node, stat, display_detail=0):
    all_mentions = event.findall(u'event_mention')
    for mention in all_mentions:
        stat[u'numEventMentions'] += 1
        role_event_mention_stat(mention, document_node, stat, display_detail)
        
def role_event_mention_stat(event_mention, document_node, stat, display_detail=0):
    anchor = event_mention.find(u'anchor')
    event_mention_args = event_mention.findall(u'event_mention_argument')
    entity_mention_ids = [x.attrib[u'REFID'] for x in event_mention_args]
    ref_ids = [re.sub(u'-[0-9]*$', u'', refid) for refid in entity_mention_ids]
    ref_nodes = [document_node.find(u'*[@ID="{0}"]'.format(rid)) for rid in ref_ids]
    char_seqs = []
    for (ref, emid) in zip(ref_nodes, entity_mention_ids):
        stat[u'numRoles'] += 1
        if ref.tag == u'entity':
            char_seqs += [ref.find(u'entity_mention[@ID="{0}"]'.format(emid))[1][0]]
        elif ref.tag == u'value':
            char_seqs += [ref.find(u'value_mention[@ID="{0}"]'.format(emid))[0][0]]
        elif ref.tag == u'timex2':
            char_seqs += [ref.find(u'timex2_mention[@ID="{0}"]'.format(emid))[0][0]]
        else:
            print(u'New Tag {0}'.format(ref.tag))
            raise()
    if display_detail > 3:
        print(u' '*display_detail + u'anchor={0}'.format(anchor[0].text))
    if u' ' in anchor[0].text or u'\n' in anchor[0].text:
        stat[u'compoundAnchor'] += 1
    for marg, char_seq in zip(event_mention_args, char_seqs):
        if display_detail > 3:
            print(u' '*display_detail + u' role={0:17} text={1}'.format(marg.attrib[u'ROLE'], char_seq.text))
        if u' ' in anchor[0].text or u'\n' in anchor[0].text:
            stat[u'compoundRole'] += 1

def print_pred_details(prediction, label, text, none_class_index):
    # none_class_index = numEventSubtypes -1
    
    label_arg_max = np.argmax(label, axis=1)
    
    event_instances = label_arg_max != none_class_index
    event_index = np.argwhere(event_instances)

    pred_arg_max = np.argmax(prediction, axis=1)
    predicted_instances = pred_arg_max != none_class_index

    predicted_index = np.argwhere(predicted_instances)
    all_index = np.union1d(event_index[:,0], predicted_index[:,0])

    for i in all_index:
        is_event = i in event_index
        is_prediction = i in predicted_index
        if is_event:
            if is_prediction:
                if pred_arg_max[i] == label_arg_max[i]:
                    print(u'TE {0:5}: {1}'.format(i, text[i]))
                else:
                    print(u'TD {0:5}: {1}'.format(i, text[i]))
            else:
                print(u'FN {0:5}: {1}'.format(i, text[i]))
        else:
            if is_prediction:
                print(u'FP {0:5}: {1}'.format(i, text[i]))

def get_tokens_by_category(prediction, label, text, none_class_index, nlp=None):
    # none_class_index = numEventSubtypes -1
    
    label_arg_max = np.argmax(label, axis=1)
    
    event_instances = label_arg_max != none_class_index
    event_index = np.argwhere(event_instances)

    pred_arg_max = np.argmax(prediction, axis=1)
    predicted_instances = pred_arg_max != none_class_index

    predicted_index = np.argwhere(predicted_instances)
    # all_index = np.union1d(event_index[:,0], predicted_index[:,0])

    result = dict()
    result[u'gt'] = set() # ground truth
    result[u'te'] = set() # true event prediction
    result[u'td'] = set() # true detection
    result[u'fn'] = set() # false negative
    result[u'fp'] = set() # false positive
    result[u'tn'] = set() # true negative
    for i in xrange(label.shape[0]):
        is_event = i in event_index
        is_prediction = i in predicted_index
        if is_event:
            start = text[i].index(u' (') + 2
            end = text[i].index(u'+)')
            if nlp:
                tokens = nlp(text[i][start:end])
                token = tokens[0]
                token_str = token.lemma_
            else:
                token_str = text[i][start:end]

            result[u'gt'].add(token_str)
            if is_prediction:
                if pred_arg_max[i] == label_arg_max[i]:
                    result[u'te'].add(token_str)
                else:
                    result[u'td'].add(token_str)
            else:
                result[u'fn'].add(token_str)
        else:
            start = text[i].index(u' (') + 2
            end = text[i].index(u')')
            if nlp:
                tokens = nlp(text[i][start:end])
                token = tokens[0]
                token_str = token.lemma_
            else:
                token_str = text[i][start:end]
            
            if is_prediction:
                result[u'fp'].add(token_str)
            else:
                result[u'tn'].add(token_str)
                
    return result

def trigger_stats(train_token_by_category, test_token_by_category):
    print(u'Number test true  positive triggers NOT in training trigger set: {0}'.format(
        len(test_token_by_category[u'te'] - train_token_by_category[u'gt'])))
    print(u'Number test false positive triggers NOT in training trigger set: {0}'.format(
        len(test_token_by_category[u'fp'] - train_token_by_category[u'gt'])))
    print(u'Number test true  positive triggers in training trigger set: {0}'.format(
        len(test_token_by_category[u'te'] & train_token_by_category[u'gt'])))
    print(u'Number test false positive triggers in training trigger set: {0}'.format(
        len(test_token_by_category[u'fp'] & train_token_by_category[u'gt'])))

def find_inst_with_token(token_set, label, text, none_class_index, prediction=None):

    label_arg_max = np.argmax(label, axis=1)
    event_instances = label_arg_max != none_class_index
    event_index = np.argwhere(event_instances)

    if prediction is None:
        predicted_index = set()
    else:
        pred_arg_max = np.argmax(prediction, axis=1)
        predicted_instances = pred_arg_max != none_class_index
        predicted_index = np.argwhere(predicted_instances)

    result = []
    false_positives = []
    for i in xrange(label.shape[0]):
        is_event = i in event_index
        is_prediction = i in predicted_index
        if is_event:
            continue
        start = text[i].index(u' (') + 2
        end = text[i].index(u')')
        token_str = text[i][start:end]
        if token_str in token_set:
            result.append(i)
            if is_prediction:
                false_positives.append(i)
    return result,false_positives

                
def read_hengji_partition_lists(def_dir=config.hengji_def_dir):

    train_file = os.path.join(def_dir, u'train')
    dev_file = os.path.join(def_dir, u'dev')
    test_file = os.path.join(def_dir, u'test')
    train = read_file_prefixes(train_file)
    dev = read_file_prefixes(dev_file)
    test = read_file_prefixes(test_file)

    return {u'train':train, u'dev':dev, u'test':test}

def approximate_text_comparison(old_text, new_text):
    u"""Compares if two arrays of strings are approximately the same or not"""
    
    omax = old_text.shape[0]
    nmax = new_text.shape[0]
    o = 0
    n = 0
    delta = 0
    while (o < omax and n < nmax):
        if old_text[o] == new_text[n]:
            o += 1
            n += 1
            delta = 0
            continue
        oset = set(old_text[o].split(u' '))
        nset = set(new_text[n].split(u' '))
        union = oset | nset
        inter = oset & nset
        ratio = len(inter) / len(union)
        if ratio > 0.8:
            print(u'o {0:8} : {1}'.format(o, old_text[o]))
            print(u'n {0:8} : {1}'.format(n, new_text[n]))
            o += 1
            n += 1
            delta = 0
            continue
        print (u'n={0} ratio: {1}'.format(n, ratio))
        print(u'old {0:8} : {1}'.format(o, old_text[o]))
        print(u'new {0:8} : {1}'.format(n, new_text[n]))

        # Assume new_text has more instances
        n += 1
        delta += 1
        if delta > 80:
            break
    return(o,n)

def gen_csv_from_keras_output(filename, num_events=0, out_filename=None):
    col_name = [u'name', u'weight', u'epoch', u'epoch_max', u'events', u'prec', u'recall', u'f']
    rows = []
    rows.append(col_name)

    with open(filename, u'r') as f:
        for line in f:
            if line.startswith(u'##=='):
                name = line.split(u' ')[-1].strip()
            elif line.startswith(u'Weight='):
                weight = float(line[7:])
                segment = []
            elif line.startswith(u'Epoch '):
                epoch, epoch_max = [int(x) for x in line[6:].split(u'/')]
            elif u'val_prec' in line:
                csv_line = [name, weight, epoch, epoch_max]
                key_value_str = line[line.index(u'val_prec'):].split(u' - ')
                values = {}
                for key, value in [x.split(u': ') for x in key_value_str]:
                    if u'prec' in key:
                        values[u'prec'] = float(value)
                    if u'recall' in key:
                        values[u'recall'] = float(value)
                    if u'val_f' in key:
                        values[u'f1'] = float(value)
                csv_line.append(num_events)
                for key in [u'prec', u'recall']:
                    csv_line.append(values[key])
                if u'f1' in values:
                    csv_line.append(values[u'f1'])
                segment.append(csv_line)
            elif u'  number of events = ' in line:
                recall_denom = int(line[20:])
                if not num_events == 0:
                    for row in segment:
                        if len(row) == 8:
                            n,w,e,m,denom,prec,r,f = row
                        else:
                            n,w,e,m,denom,prec,r = row
                        recall = r * recall_denom / num_events
                        f1 = 2 / ((1/prec) + (1/recall))
                        rows.append([n,w,e,m,num_events,prec,recall,f1])
                else:
                     rows += segment
    if out_filename is not None:
        with open(out_filename, u'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)
    return rows

