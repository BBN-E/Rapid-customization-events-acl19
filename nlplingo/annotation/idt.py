# -*- coding: utf-8 -*-

import argparse
import codecs
import re
import sys

from nlplingo.text.text_theory import Document

from nlplingo.text import text_span
from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_span import Token
from nlplingo.common.utils import IntPair


# strip certain prefixes, suffixes from the mention text
# return: new_text, prefix-length-stripped, suffix-length-stripped
def strip_mention_text(text):
    prefix_length = 0
    suffix_length = 0

    if text.startswith(u'“'):
        prefix_length = len(u'“')
    elif text.startswith('"'):
        prefix_length = 1
    elif text.startswith('('):
        prefix_length = 1

    if text.endswith(u'”'):
        suffix_length = len(u'”')
    elif text.endswith('"'):
        suffix_length = 1
    elif text.endswith(u"’s"):
        suffix_length = len(u"’s")
    elif text.endswith(')'):
        suffix_length = 1

    return (text[prefix_length:len(text)-suffix_length], prefix_length, suffix_length)


def extract_sentence_annotation(text, offset):
    """offset: char offset thus far (excluding xml tags) from prior sentences."""

    start_tag = 0
    end_tag = -1
    raw_text = ''
    entity_mentions = []

    # ignore everything starting from 'REMOVED_URL'
    url_index = text.find(' REMOVED_URL', 0)
    if url_index != -1:
        text = text[0:url_index]

    start_tag = text.find('<ENAMEX', 0)
    while(start_tag != -1):
        raw_text += text[end_tag+1 : start_tag]

        end_tag = text.find('>', start_tag)
        entity_type = re.search(r' TYPE="(.*)"', text[start_tag:end_tag]).group(1)

        start_tag = text.find('</ENAMEX>', end_tag)
        mention_text = text[end_tag+1 : start_tag]

        start = offset+len(raw_text)
        end = offset+len(raw_text)+len(mention_text)
        if '-' in mention_text and entity_type.endswith('DESC'):
            print('Rejecting %s[%s], because Spacy will split the string into multiple tokens, and DESC should always be just a single word' % (entity_type, mention_text)).encode('utf-8')
        else:
            (new_mention_text, prefix_length, suffix_length) = strip_mention_text(mention_text)
            if new_mention_text != mention_text:
                print('Revising %s to %s' % (mention_text, new_mention_text)).encode('utf-8')
            id = 'm-' + str(start+prefix_length) + '-' + str(end-suffix_length)
            entity_mentions.append(EntityMention(id, IntPair(start+prefix_length, end-suffix_length), new_mention_text, entity_type))

        raw_text += mention_text

        end_tag = text.find('>', start_tag)
        start_tag = text.find('<ENAMEX', end_tag)

    raw_text += text[end_tag+1:]

    return (raw_text, entity_mentions)


def file_to_document(filepath):
    f = codecs.open(filepath, 'r', encoding='utf8')
    sentences = []

    offset = 0
    for line in f:
        (raw_text, entity_mentions) = extract_sentence_annotation(line.strip(), offset)
        sentence = text_span.to_sentence(raw_text, offset, offset + len(raw_text))
        sentence.add_annotation('ENTITY_MENTIONS', entity_mentions)
        sentences.append(sentence)

        offset += len(raw_text) + 1  # +1 to account for newline

    f.close()

    s_strings = [s.label for s in sentences]
    doc_text = "\n".join(s_strings)

    #doc_id = os.path.basename(filepath)
    doc_id = filepath
    return Document(doc_id, IntPair(0, offset-1), doc_text, sentences)


def get_entity_label(token, entity_mentions):
    start = token.start_char_offset()
    end = token.end_char_offset()

    em_label = 'O'
    for m in entity_mentions:
        m_type = m.mention_type
        m_start = m.start_char_offset()
        m_end = m.end_char_offset()

        if start == m_start:
            em_label = 'B-'+m_type
            break
        elif m_start < start and end <= m_end:
            em_label = 'I-'+m_type
            break

    return em_label

# ignore whitespace tokens
def tokens_to_conll(all_tokens, entity_mentions):
    outlines = []

    for t in all_tokens:
        t_text = t.label

        if t_text.isspace():
            continue

        if t_text == 'REMOVED_URL':
          break;

        em_label = get_entity_label(t, entity_mentions)

        # special post processing:
        # - @mentions : em_label should be 'O', t_text should be _MENTION_
        # - numbers : t_text should be _NUMBER_
        if t_text.startswith('@') and (len(t_text) >= 2):
            em_label = 'O'
            t_text = '_MENTION_'

        if is_number(t_text):
            t_text = '_NUMBER_'
        
        # use a default pos_tag of NN , if there isn't one
        pos_tag = 'NN' if t.pos_tag is None else t.pos_tag 
        outlines.append(t_text + ' ' + pos_tag + ' ' + em_label)

    return outlines


# If tokenizer is None, we will just use sentence.tokens , which were split on whitespace
# If tokenizer is not None, then it is spacy
def print_sentence_as_conll(sentence):
    entity_mentions = sentence.annotations['ENTITY_MENTIONS']
    all_tokens = sentence.tokens
    return tokens_to_conll(all_tokens, entity_mentions)


def print_spacy_sentence_as_conll(sent, entity_mentions, offset):
    all_tokens = []

    for token in sent:
        start = token.idx + offset
        end = start + len(token.text)
        all_tokens.append(Token(IntPair(start, end), token.text, token.tag_))	# token.tag_ : POS-tag

    return tokens_to_conll(all_tokens, entity_mentions)


def is_number(n):
    return n.replace('.','',1).replace('+','',1).replace('-','',1).isdigit()

def process_idt_file(doc, idt_file):
    texts = []
    offset = 0

    f = codecs.open(idt_file, 'r', encoding='utf8')
    for line in f:
        (raw_text, entity_mentions) = extract_sentence_annotation(line.strip(), offset)
        texts.append(raw_text)
        for em in entity_mentions:
            doc.add_entity_mention(em)
        offset += len(raw_text) + 1  # +1 to account for newline
    f.close()

    idt_text = '\n'.join(texts).strip()
    if idt_text != doc.text:
        raise ValueError('Texts from IDT file does not match raw text!\n<doc.text>%s</doc.text>\n<idt_text>%s</idt_text>' % (doc.text, idt_text))

    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idt')
    parser.add_argument('--text')
    parser.add_argument('--docid')

    args = parser.parse_args()

    f = codecs.open(args.text, 'r', encoding='utf-8')
    all_text = f.read()
    f.close()

    doc = Document(args.docid, all_text.strip())
    doc = process_idt_file(doc, args.idt)
    print(doc.to_string())


