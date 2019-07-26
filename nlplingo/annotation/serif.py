
import sys

try:
    import serifxml
    from serifxml import Document as serifDoc
except ImportError as error:
    print(error.__class__.__name__ + ': ' + error.message)

from nlplingo.common.utils import IntPair
from nlplingo.text.text_theory import Document as lingoDoc
from nlplingo.text.text_span import Token
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import EntityMention

def get_snippet(serif_doc, sentence_theory):
    sentence_start = sentence_theory.token_sequence[0].start_char
    sentence_end = sentence_theory.token_sequence[-1].end_char
    return serif_doc.get_original_text_substring(sentence_start, sentence_end), sentence_start, sentence_end


def to_tokens(st):
    """
    :type st: serifxml.SentenceTheory

    Returns: list[nlplingo.text.text_span.Token]
    """
    ret = []
    """:type: list[nlplingo.text.text_span.Token]"""

    root = st.parse.root
    """:type: serifxml.SynNode"""
    for i, t in enumerate(root.terminals):
        t_text = t.text
        t_start = t.start_char
        t_end = t.end_char
        t_pos_tag = t.parent.tag
        # we do a +1 because this has been the assumption in nlplingo
        ret.append(Token(IntPair(t_start, t_end+1), i, t_text, lemma=None, pos_tag=t_pos_tag))
    return ret

def add_names(st, doc):
    """
    :type st: serifxml.SentenceTheory
    :type doc: nlplingo.text.text_theory.Document
    """
    for m in st.name_theory:
        start = m.start_char
        end = m.end_char + 1
        m_exists = False
        for em in doc.entity_mentions:
            if em.start_char_offset() == start and em.end_char_offset() == end:
                m_exists = True
                break
        if not m_exists:
            em = EntityMention(m.id, IntPair(start, end), m.text, m.entity_type)
            doc.add_entity_mention(em)

def add_entity_mentions(st, s, doc):
    """
    :type st: serifxml.SentenceTheory
    :type s: nlplingo.text.text_span.Sentence
    :type doc: nlplingo.text.text_theory.Document
    """

    for m in st.mention_set:
        if m.entity_subtype != 'UNDET':
            m_type = '{}.{}'.format(m.entity_type, m.entity_subtype)
        else:
            m_type = m.entity_type

        em = EntityMention(m.id, IntPair(m.start_char, m.end_char+1), m.text, m_type)

        head = m.head
        for t in s.tokens:
            if t.start_char_offset() == head.start_char and t.end_char_offset() == (head.end_char+1):
                em.head_token = t
                break

        doc.add_entity_mention(em)


def add_value_mentions(st, s, doc):
    """
    :type st: serifxml.SentenceTheory
    :type s: nlplingo.text.text_span.Sentence
    :type doc: nlpling.text.text_theory.Document
    """

    for m in st.value_mention_set:
        em = EntityMention(m.id, IntPair(m.start_char, m.end_char+1), m.text, m.value_type)
        doc.add_entity_mention(em)


def to_lingo_doc(filepath):
    """Takes in a filepath to a SerifXML, and use its sentences, tokens, entity-mentions, value-mentions
    to construct a nlplingo.text.text_theory.Document
    Returns: nlplingo.text.text_theory.Document
    """
    serif_doc = serifxml.Document(filepath)
    """:type: serifxml.Document"""

    docid = serif_doc.docid
    lingo_doc = lingoDoc(docid)
    for st_index, sentence in enumerate(serif_doc.sentences):
        st = sentence.sentence_theories[0]
        """:type: serifxml.SentenceTheory"""
        if len(st.token_sequence) == 0:
            continue
        st_text, st_start, st_end = get_snippet(serif_doc, st)

        tokens = to_tokens(st)
        assert st_start == tokens[0].start_char_offset()
        assert (st_end+1) == tokens[-1].end_char_offset()

        s = Sentence(docid, IntPair(st_start, st_end+1), st_text, tokens, st_index)
        add_entity_mentions(st, s, lingo_doc)
        add_value_mentions(st, s, lingo_doc)
        add_names(st, lingo_doc)

        lingo_doc.add_sentence(s)
    return lingo_doc


# TODO: add lemma dict, train_test.read_doc_annotation() method
if __name__ == "__main__":

    serifxml_filepath = sys.argv[1]
    lingo_doc = to_lingo_doc(serifxml_filepath)


    # for s in lingo_doc.sentences:
    #     token_strings = []
    #     for token in s.tokens:
    #         token_strings.append('{}-{}:{}/{}'.format(token.start_char_offset(), token.end_char_offset(), token.text, token.pos_tag))
    #     print(' '.join(token_strings))

