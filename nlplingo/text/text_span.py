from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

import re

from nlplingo.common.utils import Struct
from nlplingo.common.utils import IntPair

span_types = Struct(TEXT='TEXT', SENTENCE='SENTENCE', TOKEN='TOKEN', ENTITY_MENTION='ENTITY_MENTION',
                    EVENTSPAN='EVENTSPAN', ANCHOR='ANCHOR', EVENT_ARGUMENT='EVENT_ARGUMENT')

punctuations = {'.', '?', '!', ',', ';', ':', '-', '(', ')', '{', '}', '[', ']', '<', '>', '"', "'", "`", '/', '~', '@',
                '#', '^', '&', '*', '+', '=', '_', '\\', '|'}

"""Classes here:
Span (this is abstract)

EntityMention(Span)
Anchor(Span)
EventSpan(Span)
EventArgument(Span)

Token(Span)
Sentence(Span)
"""

class IntegerPair(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def has_same_boundary(self, other):
        """
        :type other: IntegerPair
        """
        if self.start == other.start and self.end == other.end:
            return True
        else:
            return False

    def contains(self, other):
        """
        :type other: IntegerPair
        """
        if self.start <= other.start and other.end <= self.end:
            return True
        else:
            return False

    def has_overlapping_boundary(self, other):
        """
        :type other: IntegerPair
        """
        if (self.start <= other.start and other.start <= self.end) or \
                (other.start <= self.start and self.start <= other.end) or \
                self.has_same_boundary(other) or self.contains(other) or other.contains(self):
            return True
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    def to_string(self):
        return '({},{})'.format(self.start, self.end)


class Span(object):
    """An abstract class. A Span can be any sequence of text
    :int_pair: representing the start and end character offsets of this span
    :text: the text of this span
    """

    __metaclass__ = ABCMeta

    def __init__(self, int_pair, text):
        self.int_pair = IntPair(int_pair.first, int_pair.second)
        """:type: IntPair"""
        self.text = text

    def start_char_offset(self):
        return self.int_pair.first

    def end_char_offset(self):
        return self.int_pair.second

    @abstractmethod
    def span_type(self):
        """Return a string representing the type of span this is."""
        pass


class TextSpan(Span):
    """A simple span of texts"""
    def __init__(self, int_pair, text):
        Span.__init__(self, int_pair, text)

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def span_type(self):
        return span_types.TEXT

class EntityMention(Span):
    """A consecutive span representing an entity mention.
    label: the NER type, e.g. Person, Organization, GPE
    """

    time_words = ['time', 'a.m.', 'am', 'p.m.', 'pm', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year',
                  'years', 'morning', 'afternoon', 'evening', 'night', 'anniversary', 'second', 'seconds', 'minute',
                  'minutes', 'hour', 'hours', 'decade', 'decades', 'january', 'february', 'march', 'april', 'may',
                  'june', 'july', 'august', 'september', 'october', 'november', 'december', 'today', 'yesterday',
                  'tomorrow', 'past', 'future', 'present', 'jan', 'jan.', 'feb', 'feb.', 'mar', 'mar.', 'apr', 'apr.',
                  'jun', 'jun.', 'jul', 'jul.', 'aug', 'aug.', 'sept', 'sept.', 'oct', 'oct.', 'nov', 'nov.', 'dec',
                  'dec.']

    def __init__(self, id, int_pair, text, label, entity=None, mention_type=None):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""
        self.head_token = None
        """:type: nlplingo.text.text_span.Token"""
        self.entity = entity
        self.mention_type = mention_type
    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def coarse_label(self):
        if '.' in self.label:
            type = re.match(r'^(.*?)\.', self.label).group(1)   # get coarse grained NE type
        else:
            type = self.label
        return type

    @staticmethod
    def find_first_word_before(tokens, markers):
        """
        :type tokens: list[nlplingo.text.text_span.Token]
        :type markers: set[str]
        """
        for i, token in enumerate(tokens):
            if token.text.lower() in markers and i > 0:
                return tokens[i-1]
        return None

    def head(self):
        """
        Strategy for multi-word entity mentions.
        - Crime :
            (i) if there is a verb, use it as headword
            (ii) if there is 'of' or 'to', use the word before as the head
            (iii) else, use last word as head
        - Job-Title :
            (i) if there is a 'of' or 'at', use the word before as the head
            (ii) else, use last word as head
        - Numeric : use last word as head
        - Sentence : use last word as head
        - Time : look for the words in time_words (in order) and use it as the head if found. Else, use last word.
        - All other NE types:
            (i) remove any token consisting of just numbers and periods
            (ii) use last word as head

        Returns:
            nlplingo.text.text_span.Token
        """
        if self.head_token is not None:
            return self.head_token

        if self.tokens is None: # some entity mentions are not backed by tokens
            return None

        if len(self.tokens) == 1:
            return self.tokens[0]

        type = self.coarse_label()

        if type == 'Crime':
            for token in self.tokens:
                if token.pos_category() == u'VERB':
                    return token
            t = self.find_first_word_before(self.tokens, set(['of', 'to']))
            if t is not None:
                return t
            else:
                return self.tokens[-1]
        elif type == 'Job-Title':
            t = self.find_first_word_before(self.tokens, set(['of', 'at']))
            if t is not None:
                return t
            else:
                return self.tokens[-1]
        elif type == 'Numeric' or type == 'Sentence':
            return self.tokens[-1]
        elif type == 'Time':
            for w in self.time_words:
                for token in self.tokens:
                    if token.text.lower() == w:
                        return token
            return self.tokens[-1]
        else:
            for i, token in enumerate(self.tokens):
                if token.text.lower() == 'of' and i > 0:
                    return self.tokens[i-1]
                
            toks = []
            for token in self.tokens:
                if re.search(r'[a-zA-Z]', token.text) is not None:
                    toks.append(token)
            if len(toks) > 0:
                return toks[-1]
            else:
                return self.tokens[-1]

    def is_proper_noun(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'NNP' or head_pos_tag == 'NNPS'

    def is_common_noun(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'NN' or head_pos_tag == 'NNS'

    def is_adjective(self):
        head_pos_tag = self.head().pos_tag
        return head_pos_tag == 'JJ'

    def is_noun(self):
        return self.is_proper_noun() or self.is_common_noun()

    def head_pos_category(self):
        if self.is_proper_noun():
            return 'P'
        elif self.is_common_noun():
            return 'N'
        else:
            return '?'

    #def is_proper_or_common(self):
    #    if len(self.tokens) == 1:
    #        pos_tag = self.tokens[0].spacy_token.tag_
    #        # we do the following because some NNP (e.g. Chinese, Czech) are mis-tagged as JJ
    #        if pos_tag == 'PRP' or pos_tag == 'PRP$' or pos_tag == 'WP' or pos_tag == 'CD' or pos_tag == 'WDT':
    #           return False
    #    return True

    def span_type(self):
        return span_types.ENTITY_MENTION

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))


class Anchor(Span):
    """A consecutive span representing an anchor
    label: the event type represented by the anchor, e.g. Conflict.Attack, CyberAttack, Vulnerability
    """

    def __init__(self, id, int_pair, text, label):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def head(self):
        """If the anchor is just a single token, we return the single token.
        If the anchor is multi-words, we heuristically determine a single token as the head

        Returns:
            nlplingo.text.text_span.Token
        """
        if len(self.tokens) == 1:
            return self.tokens[0]
        else:
            if self.tokens[0].pos_category() == u'VERB':
                return self.tokens[0]
            else:
                return self.tokens[-1]

    def span_type(self):
        return span_types.ANCHOR

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))


class EventSpan(Span):
    """A consecutive span representing an event. Sometimes we explicitly label e.g. a sentence as the event span.
    label: the event type represented by the event, e.g. Conflict.Attack, CyberAttack, Vulnerability
    """

    def __init__(self, id, int_pair, text, label):
        Span.__init__(self, int_pair, text)
        self.id = id
        self.label = label
        self.tokens = None
        """:type: list[nlplingo.text.text_span.Token]"""
        self.sentences = None

    def with_tokens(self, tokens):
        """:type tokens: list[nlplingo.text.text_span.Token]"""
        self.tokens = tokens

    def with_sentences(self, sentences):
        """:type sentences: list[nlplingo.text.text_span.Sentence]"""
        self.sentences = sentences

    # def start_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[0].start_char_offset()
    #     else:
    #         return self.int_pair.first
    #
    # def end_char_offset(self):
    #     if self.tokens is not None:
    #         return self.tokens[-1].end_char_offset()
    #     else:
    #         return self.int_pair.second

    def span_type(self):
        return span_types.EVENTSPAN

    def to_string(self):
        return (u'%s: %s (%d,%d) "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, self.label))


class EventArgument(Span):
    """A consecutive span representing an event argument
    label: the event argument role, e.g. Source, Target
    """

    def __init__(self, id, entity_mention, label):
        """:type entity_mention: nlplingo.text.text_span.EntityMention"""
        Span.__init__(self, IntPair(entity_mention.start_char_offset(), entity_mention.end_char_offset()), entity_mention.text)
        self.id = id
        self.label = label
        self.entity_mention = entity_mention

    def copy_with_entity_mention(self, entity_mention):
        """Sometimes we want to reassign the entity_mention with one that is backed by tokens
        :type entity_mention: nlplingo.text.text_span.EntityMention"""
        return EventArgument(self.id, entity_mention, self.label)

    def span_type(self):
        return span_types.EVENT_ARGUMENT

    def to_string(self):
        if self.entity_mention.tokens is not None:
            postags = ' '.join(token.pos_tag for token in self.entity_mention.tokens)
        else:
            postags = 'N.A.'
        return (u'%s: %s (%d,%d) "%s" "%s" %s' % (self.span_type(), self.id, self.start_char_offset(), self.end_char_offset(), self.text, postags, self.label))



class Token(Span):
    """An individual word token.
    :spacy_token: an optional spacy token
    """

    def __init__(self, int_pair, index, text, lemma, pos_tag):
        Span.__init__(self, int_pair, text)
        self.index_in_sentence = index     # token index in sentence

        self.lemma = lemma
        self.pos_tag = pos_tag
        self.dep_relations = []             # dependency relations
        """:type: list[nlplingo.text.dependency_relation.DependencyRelation]"""
        self.child_dep_relations = []
        """:type: list[nlplingo.text.text_span.DependencyRelation]"""

        self.dep_paths_to_root = []
        """:type: list[list[nlplingo.text.text_span.DependencyRelation]]"""
        # might not be completely to root, as we impose a max path length

        # following deals with word embeddings
        self.has_vector = False
        self.vector_index = 0
        self.word_vector = None

        self.srl = None
        """:type: nlplingo.text.text_theory.SRL"""

    def add_dep_relation(self, dep_relation):
        self.dep_relations.append(dep_relation)

    def add_child_dep_relation(self, dep_relation):
        self.child_dep_relations.append(dep_relation)

    def add_dep_path_to_root(self, path):
        self.dep_paths_to_root.append(path)

    def is_punct(self):
        return self.text in punctuations

    def pos_suffix(self):
        if self.pos_tag.startswith('NN'):
            return '.n'
        elif self.pos_tag.startswith('VB'):
            return '.v'
        elif self.pos_tag.startswith('JJ'):
            return '.a'
        else:
            return '.o'

    def text_with_pos_suffix(self):
        return self.text + self.pos_suffix()

    def pos_category(self):
        if self.pos_tag.startswith('NNP'):
            return 'PROPN'
        elif self.pos_tag.startswith('NN'):
            return 'NOUN'
        elif self.pos_tag.startswith('VB'):
            return 'VERB'
        elif self.pos_tag.startswith('JJ'):
            return 'ADJ'
        else:
            return 'OTHER'

    def span_type(self):
        return span_types.TOKEN

    def to_string(self):
        return (u'%s: (%d,%d) "%s"' % (self.span_type(), self.start_char_offset(), self.end_char_offset(), self.text))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start_char_offset() == other.start_char_offset() and self.end_char_offset() == other.end_char_offset()
        return False

    def __ne__(self, other):
        return self.start_char_offset() != other.start_char_offset() or self.end_char_offset() != other.end_char_offset()

class Sentence(Span):
    """Represents a sentence
    :tokens: a list of Token
    """

    def __init__(self, docid, int_pair, text, tokens, index):
        """:index: int giving the sentence number (starting from 0) within the document"""
        Span.__init__(self, int_pair, text)
        self.docid = docid
        self.tokens = tokens
        """:type: list[nlplingo.text.text_span.Token]"""
        self.noun_phrases = self._add_noun_phrases()
        """:type: list[nlplingo.text.text_span.TextSpan]"""
        self.entity_mentions = []
        """:type: list[nlplingo.text.text_span.EntityMention]"""
        self.events = []
        """:type: list[nlplingo.text.text_theory.Event]"""
        self.srls = []
        """:type: list[nlplingo.text.text_theory.SRL]"""
        self.index = index
        self.sent_id = None
        self.vector_index = -1
        self.sent_vector = []
        self.has_vector = False

    def _add_noun_phrases(self):
        """Now, let's just add all bigrams and trigrams
        """
        ret = []
        """:type: list[nlplingo.text.text_span.TextSpan]"""
        for i in range(len(self.tokens) - 1):  # bigrams
            toks = self.tokens[i:i + 2]
            span = TextSpan(IntPair(toks[0].start_char_offset(), toks[-1].end_char_offset()), ' '.join(t.text for t in toks))
            span.with_tokens(toks)
            ret.append(span)
        for i in range(len(self.tokens) - 2):  # trigrams
            toks = self.tokens[i:i + 3]
            span = TextSpan(IntPair(toks[0].start_char_offset(), toks[-1].end_char_offset()), ' '.join(t.text for t in toks))
            span.with_tokens(toks)
            ret.append(span)
        return ret

    def add_entity_mention(self, entity_mention):
        """:type entity_mention: nlplingo.text.text_span.EntityMention"""
        self.entity_mentions.append(entity_mention)

    def add_event(self, event):
        """:type event: nlplingo.text.text_theory.Event"""
        self.events.append(event)

    def add_srl(self, srl):
        """:type srl: nlplingo.text.text_theory.SRL"""
        self.srls.append(srl)

    def number_of_tokens(self):
        return len(self.tokens)

    def span_type(self):
        return span_types.SENTENCE

    def get_all_event_anchors(self):
        """Returns a list of all event anchors
        Returns:
            list[nlplingo.text.text_span.Anchor]
        """
        # TODO: when reading in events from the annotation files, ensure each token/anchor is only used once
        ret = []
        for event in self.events:
            for anchor in event.anchors:
                ret.append(anchor)
        return ret

    def get_ne_type_per_token(self):
        ret = []
        for token in self.tokens:
            found_ne = False
            for em in self.entity_mentions:
                if em.start_char_offset() <= token.start_char_offset() and token.end_char_offset() <= em.end_char_offset():
                #if token.index_in_sentence == em.head().index_in_sentence:
                    ret.append(em.coarse_label())
                    found_ne = True
                    break
            if not found_ne:
                ret.append('None')
        assert len(ret) == len(self.tokens)
        return ret

    def get_ne_type_with_bio_per_token(self):
        ret = []
        for token in self.tokens:
            found_bio = False
            for em in self.entity_mentions:
                for i, em_token in enumerate(em.tokens):  # type: Token
                    if em_token.index_in_sentence == token.index_in_sentence:
                        if i == 0:
                            label = em.coarse_label() + "_B"
                        else:
                            label = em.coarse_label() + "_I"
                        ret.append(label)
                        found_bio = True
                        break
                if found_bio:
                    break
            if not found_bio:
                ret.append('O')
        assert len(ret) == len(self.tokens)
        return ret

    def get_text(self, start, end):
        """If the given start, end character offsets are within this sentence, return the associated text"""
        if self.start_char_offset() <= start and end <= self.end_char_offset():
            normalized_start = start - self.start_char_offset()
            normalized_end = normalized_start + (end - start)
            #print('Returning {}-{} from "{}"'.format(normalized_start, normalized_end, self.text))
            return self.text[normalized_start:normalized_end]
        else:
            return None

    def to_string(self):
        return u'(%d,%d):[%s]' % (self.start_char_offset(), self.end_char_offset(), self.text)

#
# class DependencyRelation(object):
#     def __init__(self, dep_name, parent_token_index, child_token_index):
#         self.dep_name = dep_name
#         self.parent_token_index = parent_token_index
#         self.child_token_index = child_token_index
#
#     def to_string(self):
#         return '{}:{}:{}'.format(self.dep_name, self.parent_token_index, self.child_token_index)



def to_sentence(text, start, end):
    """Converts a sentence raw text to a Sentence object."""
    charOffsets = IntPair(start, end)
    tokens = []

    offset = start
    for t in text.split():
        token = Token(IntPair(offset, offset+len(t)), t)
        tokens.append(token)
        offset += len(t)+1    # +1 to account for white-space

    return Sentence(charOffsets, text, tokens)

def file_to_document(filepath):
    f = open(filepath, 'rU')
    sentences = []

    offset = 0
    for line in f:
        sentence = to_sentence(line, offset, offset+len(line))
        sentences.append(sentence)
        offset += len(line);    # +1 for account for newline
    f.close()

    s_strings = [s.label for s in sentences]
    doc_text = "\n".join(s_strings)

    return Document(IntPair(0, offset-1), doc_text, sentences)


def spans_overlap(span1, span2):
    """
    :type span1: nlplingo.text.text_span.Span
    :type span2: nlplingo.text.text_span.Span
    """
    start1 = span1.start_char_offset()
    end1 = span1.end_char_offset()
    start2 = span2.start_char_offset()
    end2 = span2.end_char_offset()

    if start1 != start2 and end1 != end2 and (end1 <= start2 or end2 <= start1):
        return False
    else:
        return True

