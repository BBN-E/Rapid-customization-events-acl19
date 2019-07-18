from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import xml.etree.ElementTree as etree

import numpy as np
from future.builtins import zip

import config
from ace import ace_utils
from corpus.corpus import Corpus, CorpusDocument, Annotation
from event_domain import AceDomain
from event_domain import Entity, EntityMention, Time, TimeMention, Value, ValueMention
from event_domain import Event, EventMention, Argument, ArgumentMention
from spacy_wrapper import DocWrapper as Doc

## constants

# File suffix
AFP_SUFFIX = '.apf.xml'
SGML_SUFFIX = '.sgm'
    
# XML elements
EVENT = 'event'
EVENT_ARGUMENT = 'event_argument'
EVENT_MENTION = 'event_mention'
EVENT_MENTION_ARGUMENT = 'event_mention_argument'
ENTITY = 'entity'
ENTITY_MENTION = 'entity_mention'
ANCHOR = 'anchor'
HEAD = 'head'
TIMEX2 = 'timex2'
TIMEX2_MENTION = 'timex2_mention'
VALUE = 'value'
VALUE_MENTION = 'value_mention'

# XML attributes
ID = 'ID'
REFID = 'REFID'
TYPE = 'TYPE'
SUBTYPE = 'SUBTYPE'
ROLE = 'ROLE'
START = 'START'
END = 'END'

verbose = 0
class AceCorpus(Corpus):
    """Top-level class for ACE 2005 corpus"""

    NAME = 'ACE2005'

    def __init__(self, word_embedding='baroni', use_bio_index=False, replace_newline_with_space=True,
                 extract_one_word_head=True, ace_data_dir=config.ace_data_dir):
        print("\nace_docs.py : AceCorpus.__init__()")

        super(AceCorpus, self).__init__('ACE2005', word_embedding=word_embedding)
        self.domain = AceDomain()
        self.use_bio_index = use_bio_index
        self.replace_newline_with_space = replace_newline_with_space
        self.extract_one_word_head= extract_one_word_head
        self.ace_data_dir = ace_data_dir
        self.lookup = None

        # If true, then create compound token for multi-token arguments. This is used during training.
        # If false, then find nearest neighbor word for multi-token arguments. This is used during prediction.
        self.create_word = True
        self.compound_words = set()

    def get_docid_path(self, docid):
        if self.lookup is None:
            self.lookup = dict()
            for dir_name, subdir_list, filelist in os.walk(self.ace_data_dir):
                docids = [f[:-8] for f in filelist if f.endswith(AFP_SUFFIX)]
                for did in docids:
                    self.lookup[did] = os.path.join(os.path.relpath(dir_name, self.ace_data_dir), did)
        doc_path = self.lookup[docid]
        return doc_path

    def get_document(self, docid):
        doc_path = self.get_docid_path(docid)
        xml_filename = doc_path + AFP_SUFFIX
        sgml_filename = doc_path + SGML_SUFFIX
        return AceDocument(docid, self, xml_filename, sgml_filename)

    def get_document_by_path(self, doc_path):
        xml_filename = doc_path + AFP_SUFFIX
        sgml_filename = doc_path + SGML_SUFFIX
        docid = doc_path.split('/')[-1]
        return AceDocument(docid, self, xml_filename, sgml_filename)

    def set_training_mode(self):
        super(AceCorpus, self).set_training_mode()
        self.create_word = True

    def set_prediction_mode(self):
        super(AceCorpus, self).set_prediction_mode()
        self.create_word = False

    def get_num_of_events(self):
        return 424

    def get_num_of_roles(self):
        return 899
        
class AceDocument(CorpusDocument):
    def __init__(self, docid, corpus, xml_file, sgml_file, batch_size=10, n_threads=8):
        super(AceDocument, self).__init__(docid, corpus)
        self.xml_file = xml_file
        self.sgml_file = sgml_file
        self.sgml_tree = etree.parse(os.path.join(corpus.ace_data_dir, sgml_file))
        self.sgml_root = self.sgml_tree.getroot()
        self.text_list = ace_utils.extract_text(self.sgml_root, replace_newline_with_space=corpus.replace_newline_with_space)

        self.tree = etree.parse(os.path.join(corpus.ace_data_dir, xml_file))
        self.root = self.tree.getroot()
        self.document_node = self.root[0]

        self.batch_size = batch_size
        self.n_threads = n_threads

        self.events = []
        self.entities = []
        self.times = []
        self.values = []

        self.process_text()
        self.process_tokens()
        self.process_sgml()

        self.annotation = AceAnnotation(self, create_word=self.corpus.create_word)

    def get_text_units(self):
        return self.sentences

    def process_text(self):
        """Parse each string in text_list using spaCy.
        Returns parsed docs with each doc's location index."""

        
        self.all_text = ''.join(self.text_list)
        self.doc_list = []
        self.doc_psum = [0]
        self.sentences = []
        self.sent_text_idx = []

        for i, doc in enumerate(self.corpus.nlp.pipe(self.text_list, batch_size=self.batch_size,
                                                     n_threads=self.n_threads)):
            self.doc_list.append(Doc(doc, i))
            self.doc_psum.append(self.doc_psum[-1] + len(self.text_list[i]))
        for doc, idx in zip(self.doc_list, self.doc_psum):
            for span in doc.sents:
                self.sentences.append(span)
                sent_range = (idx+doc[span.start].idx, idx+doc[span.end-1].idx)
                self.sent_text_idx.append(sent_range)
        for i,sent_span in enumerate(self.sentences):
            for token in sent_span:
                token.sent_idx = i
                token.text_unit_idx = i

    def process_sgml(self):
        self.process_events()
        self.process_entities()
        self.process_times()
        self.process_values()

    def process_events(self):
        for event_node in self.document_node.findall(EVENT):
            event_id = event_node.attrib[ID]
            event_type = event_node.attrib[TYPE]
            event_subtype = event_node.attrib[SUBTYPE]
            event = Event(event_id, (event_type, event_subtype))
            if verbose:
                print('Event id={} type={} subtype={}'.format(event_id, event_type, event_subtype))
            self.events.append(event)
            for event_argument_node in event_node.findall(EVENT_ARGUMENT):
                argument = Argument(event_argument_node.attrib[REFID], event_argument_node.attrib[ROLE])
                if verbose:
                    print(' Argument refid={} role={}'.format(event_argument_node.attrib[REFID],
                                                              event_argument_node.attrib[ROLE]))
                event.add_argument(argument)

            for mention_node in event_node.findall(EVENT_MENTION):
                mention_id = mention_node.attrib[ID]
                anchor = mention_node.find(ANCHOR)
                charseq = anchor[0]
                text = charseq.text
                if self.corpus.replace_newline_with_space:
                    text = text.replace('\n', ' ')
                start = int(charseq.attrib[START])
                end = int(charseq.attrib[END])
                event_mention = EventMention(mention_id, event)
                event_mention.set_text(text, start, end)
                event.add_mention(event_mention)
                if verbose:
                    print(' EventMention id={} text={}'.format(mention_id, text))
                      
                for argument_mention_node in mention_node.findall(EVENT_MENTION_ARGUMENT):
                    argument_mention = ArgumentMention(argument_mention_node.attrib[REFID],
                                                      argument_mention_node.attrib[ROLE])
                    if verbose:
                        print(' ArgumentMention refid={} role={}'.format(argument_mention_node.attrib[REFID],
                                                                         argument_mention_node.attrib[ROLE]))
                    event_mention.add_argument_mention(argument_mention)

    def process_entities(self):
        all_entities = self.document_node.findall(ENTITY)
        for entity_node in all_entities:
            entity_id = entity_node.attrib[ID]
            entity_type = entity_node.attrib[TYPE]
            entity_subtype = entity_node.attrib[SUBTYPE]
            entity = Entity(entity_id, (entity_type, entity_subtype))
            self.entities.append(entity)
            
            all_mentions = entity_node.findall(ENTITY_MENTION)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib[ID]
                head = mention_node.find(HEAD)
                charseq = head[0]
                text = charseq.text
                if self.corpus.replace_newline_with_space:
                    text = text.replace('\n', ' ')
                start = int(charseq.attrib[START])
                end = int(charseq.attrib[END])
                mention = EntityMention(mention_id, entity)
                mention.set_text(text, start, end)
                entity.add_mention(mention)

    def process_times(self):
        for time_node in self.document_node.findall(TIMEX2):
            time_id = time_node.attrib[ID]
            time = Time(time_id, 'Time')
            self.times.append(time)
            all_mentions = time_node.findall(TIMEX2_MENTION)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib[ID]
                charseq = mention_node[0][0]
                text = charseq.text
                if self.corpus.replace_newline_with_space:
                    text = text.replace('\n', ' ')
                start = int(charseq.attrib[START])
                end = int(charseq.attrib[END])
                mention = TimeMention(mention_id, time)
                mention.set_text(text, start, end)
                time.add_mention(mention)

    def process_values(self):
        for value_node in self.document_node.findall(VALUE):
            value_id = value_node.attrib[ID]
            value_type = value_node.attrib[TYPE]
            value = Value(value_id, value_type)
            self.values.append(value)
            all_mentions = value_node.findall(VALUE_MENTION)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib[ID]
                charseq = mention_node[0][0]
                text = charseq.text
                if self.corpus.replace_newline_with_space:
                    text = text.replace('\n', ' ')
                start = int(charseq.attrib[START])
                end = int(charseq.attrib[END])
                mention = ValueMention(mention_id, value)
                mention.set_text(text, start, end)
                value.add_mention(mention)

class AceAnnotation(Annotation):
    """Annotate tokens with information from apf.xml file"""

    TIME_WORDS = [
        'time', 'a.m.', 'am', 'p.m.', 'pm', 'A.M.', 'AM', 'P.M.', 'PM', 'day',
        'days', 'week', 'weeks', 'month', 'months', 'year', 'years', 'morning',
        'afternoon', 'evening', 'night', 'anniversary', 'birtday', 'second',
        'seconds', 'minute', 'minutes', 'hour', 'hours', 'decade', 'decades',
        'era', 'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December', 'today',
        'yesterday', 'tomorrow', 'past', 'future', 'present', 'Jan', 'Jan.',
        'Feb', 'Feb.', 'Mar', 'Mar.', 'Apr', 'Apr.', 'Jun', 'Jun.', 'Jul',
        'Jul.', 'Aug', 'Aug.', 'Sept', 'Sept.', 'Oct', 'Oct.', 'Nov', 'Nov.',
        'Dec', 'Dec.', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday', 'january', 'february', 'march', 'april', 'may',
        'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan.', 'feb', 'feb.', 'mar', 'mar.', 'apr', 'apr.', 'jun', 'jun.',
        'jul', 'jul.', 'aug', 'aug.', 'sept', 'sept.', 'oct', 'oct.', 'nov',
        'nov.', 'dec', 'dec.', 'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday']
    TIME_WORD_SET = set(TIME_WORDS)
    
    def __init__(self, ace_document, create_word=True):
        super(AceAnnotation, self).__init__(ace_document)
        # self.word_embedding = ace_document.corpus.word_embedding
        self.use_bio_index = ace_document.corpus.use_bio_index
        self.extract_one_word_head = ace_document.corpus.extract_one_word_head
        
        self._tag_token_lookup = dict()

        self.of_token = self.document.corpus.nlp('of')[0]
        self.to_token = self.document.corpus.nlp('to')[0]
        self.at_token = self.document.corpus.nlp('at')[0]

        self._annotate_entity(create_word=create_word)
        self._annotate_time()
        self._annotate_value()
        self._annotate_trigger()

    def find_all(self, tokens, tag_id):
        """returns subsequence of 'tokens' with 'tag_id'"""
        result = []
        for i, tk in enumerate(tokens):
            if tk and tag_id in tk.tags:
                result.append(i)
        return result
            
    def _annotate_bio(self, mention, bio_type):
        b = ace_utils.get_bio_index(bio_type, True)
        i = ace_utils.get_bio_index(bio_type, False)
        text = mention.text
        if self.document.corpus.replace_newline_with_space:
            text = text.replace('\n', ' ')
        tokens = self.document.find_tokens(text, mention.start, mention.end)
        first = True
        for tk in tokens:
            if tk.pos_ == 'PUNCT' or tk.pos_ == 'SPACE':
                continue
            if first:
                first = False
                tk.bio_index = b
            else:
                tk.bio_index = i
        if self.document.corpus.get_word_embedding() is not None:
            for tk in tokens:
                if tk.pos_ == 'PUNCT' or tk.pos_ == 'SPACE':
                    continue
                if not tk.has_vector and self.use_bio_index:
                    bio = ace_utils.BIO_ENTITY_TYPE[tk.bio_index]
                    tk.vector_text, tk.vector_index, tk.word_vector = self.document.corpus.get_word_embedding().get_vector(bio)
                    tk.has_vector = True
                    tk.skip = False

        return tokens

    def _set_entity_head(self, tokens, create_word=True):
        head_token = tokens[-1]
        head_text = '_'.join([tk.text for tk in tokens])
        head_text_cap =  '_'.join([tk.text.capitalize() for tk in tokens])
        seen_previously = head_text in self.document.corpus.compound_words
        _, idx, head_vector = self.document.corpus.get_word_embedding().get_vector(head_text)
        if idx < 0:
            # Try capitalize words, since these are mostly proper nouns
            _, idx, head_vector = self.document.corpus.get_word_embedding().get_vector(head_text_cap)
            if idx >= 0:
                head_text = head_text_cap
                seen_previously = head_text in self.document.corpus.compound_words
                if verbose > 0 and not seen_previously:
                    print('Found capitalized compound_words: {}'.format(head_text))
        if verbose > 0 and not seen_previously and idx >=0:
            print('Embedding already has compound word: {}'.format(head_text))
        self.document.corpus.compound_words.add(head_text)
        if idx < 0:
            if create_word:
                _, idx, head_vector = self._add_word(head_text, tokens)
            else:
                _, idx, head_vector = self._find_nearest_neighbor(head_text, tokens)
        if idx < 0:
            if create_word:
                _, idx, head_vector = self._add_word(head_text_cap, tokens)
            else:
                _, idx, head_vector = self._find_nearest_neighbor(head_text_cap, tokens)
            if idx >= 0:
                head_text = head_text_cap
        if idx < 0:
            if create_word:
                print('Failed to create new word: {}'.format(head_text))
            else:
                print('Failed to find neighbor word: {}'.format(head_text))
        else:
            head_token.vector_text = head_text
            head_token.has_vector = True
            head_token.word_vector = head_vector
        return head_token

    def _add_word(self, text, tokens):
        vec, count = self._compute_vector(tokens)
        idx = -1
        if count > 0:
            idx = self.document.corpus.get_word_embedding().add_vector(text, vec)
        return (text, idx, vec)

    def _find_nearest_neighbor(self, text, tokens):
        if text in self.document.corpus.get_word_embedding().get_token_map():
            source_token = self.document.corpus.get_word_embedding().get_token_map()[text]
            if source_token:
                return self.document.corpus.get_word_embedding().get_vector(source_token)
            else:
                return (text, -1, None)
        vec, count = self._compute_vector(tokens)
        if count > 0:
            dist, idx = self.document.corpus.get_word_embedding().approximate_nearest_neighbors(vec, num_neighbors=1)
            print ('neighbor of {} is {}'.format(text, self.document.corpus.get_word_embedding().words[idx[0,0]]))
            self.document.corpus.get_word_embedding().add_token_map_entry(
                text, self.document.corpus.get_word_embedding().words[idx[0,0]])
            return (text, idx[0,0], self.document.corpus.get_word_embedding().word_vec[idx[0,0]])
        else:
            self.document.corpus.get_word_embedding().add_token_map_entry(text, None)
            return (text, -1, None)
        

    def _compute_vector(self, tokens):
        vec = np.zeros(self.document.corpus.get_word_embedding().vector_length)
        count = 0
        for i, tk in enumerate(tokens):
            if tk.pos_ == u'PUNCT' or tk.pos_ == u'SPACE':
                continue
            if tk.has_vector:
                vec += tk.word_vector
                count += 1
        if count > 0:
            vec = vec / count
        return (vec, count)

    def _annotate_entity(self, create_word=True):
        for entity in self.document.entities:
            for mention in entity.mentions:
                tokens = self.document.find_tokens(mention.text, mention.start, mention.end)
                if len(tokens) > 1:
                    head = self._set_entity_head(tokens, create_word=create_word)
                    if verbose > 1:
                        print('Entity: {} <- {}'.format(head.vector_text, [tk.text for tk in tokens]))
                    tokens = self._annotate_tokens([head], mention.id)
                else:
                    tokens = self._annotate_tokens(tokens, mention.id)
                sent_indices = set(tk.sent_idx for tk in tokens)
                if len(sent_indices) > 1:
                    print('Value tokens with ID={} cross sentence boundary: {}'.format(
                        mention.id, tokens))
                for sent_idx in sent_indices:
                    self.append_entity_mention(sent_idx, mention)
                # tag tokens with entity type (entity.type[0]), not subtype
                self._annotate_bio(mention, entity.type[0])

    def _get_time_head(self, tokens):
        token_text = [tk.text for tk in tokens]
        token_text_set = set(token_text)
        if len(tokens) == 1:
            return tokens[0]
        if self.TIME_WORD_SET.isdisjoint(token_text_set):
            return tokens[-1]
        for word in self.TIME_WORDS:
            if word in token_text_set:
                index = token_text.index(word)
                return tokens[index]

    def _annotate_time(self):
        for time in self.document.times:
            for mention in time.mentions:
                # tag tokens with entity type, entity.type[0]
                tokens = self.document.find_tokens(mention.text, mention.start, mention.end)
                if len(tokens) > 1:
                    head = self._get_time_head(tokens)
                    if verbose > 1:
                        print('Time:   {} <- {}'.format(head.text, [tk.text for tk in tokens]))
                    tokens = self._annotate_tokens([head], mention.id)
                else:
                    tokens = self._annotate_tokens(tokens, mention.id)
                sent_indices = set(tk.sent_idx for tk in tokens)
                if len(sent_indices) > 1:
                    print('Value tokens with ID={} cross sentence boundary: {}'.format(
                        mention.id, tokens))
                for sent_idx in sent_indices:
                    self.append_entity_mention(sent_idx, mention)
                self._annotate_bio(mention, time.type)

    def _get_value_head(self, value_type, tokens):
        if value_type == 'Crime':
            head = [tk for tk in tokens if tk.pos_=='VERB']
            if head:
                return(head[0])
            for i, tk in enumerate(tokens):
                if i > 0 and (tk.text == self.of_token.text or tk.text == self.to_token.text):
                    return(tokens[i-1])
        elif value_type == 'Job-Title':
            for i, tk in enumerate(tokens):
                if i > 0 and (tk.text == self.of_token.text or tk.text == self.at_token.text):
                    return(tokens[i-1])
        return(tokens[-1])
        
    def _annotate_value(self):
        for value in self.document.values:
            for mention in value.mentions:
                # tag tokens with entity type, entity.type[0] 
                tokens = self.document.find_tokens(mention.text, mention.start, mention.end)
                if len(tokens) > 1:
                    head = self._get_value_head(value.type, tokens)
                    if verbose > 1:
                        print('Value:  {} <- {}'.format(head.text, [tk.text for tk in tokens]))
                    tokens = self._annotate_tokens([head], mention.id)
                else:
                    tokens = self._annotate_tokens(tokens, mention.id)
                sent_indices = set(tk.sent_idx for tk in tokens)
                if len(sent_indices) > 1:
                    print('Value tokens with ID={} cross sentence boundary: {}'.format(
                        mention.id, tokens))
                for sent_idx in sent_indices:
                    self.append_entity_mention(sent_idx, mention)
                self._annotate_bio(mention, value.type)
        
    def _annotate_trigger(self):
        for event in self.document.events:
            for mention in event.mentions:
                tokens = self._annotate_mention(mention)
                sent_indices = set(tk.sent_idx for tk in tokens)
                if len(sent_indices) > 1:
                    raise Exception('In {0} tokens cross sentence boundary: {1}'.format(
                        mention.id, tokens))
                for sent_idx in sent_indices:
                    self.append_event_mention(sent_idx, mention)

