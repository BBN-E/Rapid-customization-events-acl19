from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import codecs
import re
import xml.etree.ElementTree as etree

from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import Event
from nlplingo.text.text_theory import Entity

from nlplingo.text.text_span import EntityMention

from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import EventArgument
from nlplingo.common import utils
from nlplingo.common.utils import IntPair

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

class AceAnnotation(object):

    @staticmethod
    def parse_filelist_line(line):
        text_file = None
        xml_file = None

        for file in line.strip().split():
            if file.startswith('TEXT:'):
                text_file = file[len('TEXT:'):]
            elif file.startswith('APF:'):
                xml_file = file[len('XML:'):]

        if text_file is None:
            raise ValueError('text file must be present!')
        return (text_file, xml_file)

    @classmethod
    def read_docs(cls, filelists):
        """
        :type filelists: list[str]
        Returns:
            list[nlplingo.text.text_theory.Document]
        """
        docs = []
        """:type docs: list[text.text_theory.Document]"""
        for line in filelists:
            (text_file, xml_file) = cls.parse_filelist_line(line)

            docid = re.match(r'(.*).sgm', os.path.basename(text_file)).group(1)
            with codecs.open(text_file, 'r', encoding='utf-8') as f:
                all_text = f.read()
            doc = Document(docid, all_text.strip())

            if xml_file is not None:
                cls.process_ace_xmlfile(doc, xml_file)  # adds entity mentions, times, values, events

            docs.append(doc)
        return docs

    @classmethod
    def extract_text(cls, node, all_text):
        u""" Return list of text from XML element subtree.
        Python 2 version"""
        tag = node.tag
        if not isinstance(tag, str) and tag is not None:
            return
        text = node.text
        if text:
            #print('A:{}[{}]'.format(str(len(text)), str(text)))
            #text = text.replace('\n', ' ')
            all_text.append(str(text))
        for e in node:
            cls.extract_text(e, all_text)
            text = e.tail
            if text:
                #print('B:{}[{}]'.format(str(len(text)), text))
                #text = text.replace('\n', ' ')
                all_text.append(text)
        return all_text

    @classmethod
    def process_ace_textfile_to_list(cls, text_file):
        tree = etree.parse(text_file)
        root_node = tree.getroot()
        return cls.extract_text(root_node, [])

    @classmethod
    def process_ace_xmlfile(cls, doc, xml_file):
        """
        :type doc: nlplingo.text.text_theory.Document
        :type xml_file: str
        Returns:
            nlplingo.text.text_theory.Document
        """
        tree = etree.parse(xml_file)
        root_node = tree.getroot()
        document_node = root_node[0]

        cls.process_entities(doc, document_node)
        cls.process_times(doc, document_node)
        cls.process_values(doc, document_node)
        cls.process_events(doc, document_node)
        return doc

    @staticmethod
    def process_xml_charseq(charseq):
        """
        :type charseq: xml.etree.ElementTree.Element
        :return: (str, int, int)
        """
        text = charseq.text.replace('\n', ' ')
        start = int(charseq.attrib['START'])
        end = int(charseq.attrib['END']) + 1
        return (text, start, end)

    @classmethod
    def process_events(cls, doc, document_node):
        """
        :type doc: nlplingo.text.text_theory.Document
        :type document_node: xml.etree.ElementTree.Element
        """
        for event_node in document_node.findall('event'):
            event_id = event_node.attrib['ID']
            event_type = event_node.attrib['TYPE']
            event_subtype = event_node.attrib['SUBTYPE']
            #for event_argument_node in event_node.findall('event_argument'):
            #    argument = Argument(event_argument_node.attrib['REFID'], event_argument_node.attrib['ROLE'])
            #    event.add_argument(argument)

            for mention_node in event_node.findall('event_mention'):
                mention_id = mention_node.attrib['ID']
                event = Event(mention_id, event_type+'.'+event_subtype)

                anchor = mention_node.find('anchor')
                (text, start, end) = cls.process_xml_charseq(anchor[0])
                event.add_anchor(Anchor(mention_id+'-trigger', IntPair(start, end), text, event_type+'.'+event_subtype))

                for argument_mention_node in mention_node.findall('event_mention_argument'):
                    arg_id = argument_mention_node.attrib['REFID']
                    arg_role = argument_mention_node.attrib['ROLE']
                    arg_em = doc.get_entity_mention_with_id(arg_id)
                    assert arg_em is not None

                    if arg_role.startswith('Time-'):
                        arg_role = 'Time'

                    event_arg = EventArgument('{}-a{}'.format(mention_id, event.number_of_arguments()), arg_em, arg_role)
                    event.add_argument(event_arg)
                doc.add_event(event)

    @classmethod
    def process_entities(cls, doc, document_node):
        """
        :type doc: nlplingo.text.text_theory.Document
        :type document_node: xml.etree.ElementTree.Element
        """
        all_entities = document_node.findall('entity')
        for entity_node in all_entities:
            entity_id = entity_node.attrib['ID']
            entity_type = entity_node.attrib['TYPE']
            entity_subtype = entity_node.attrib['SUBTYPE']

            all_mentions = entity_node.findall('entity_mention')
            entity = Entity(entity_id)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib['ID']
                mention_type = mention_node.attrib['TYPE']
                head = mention_node.find('head')
                (text, start, end) = cls.process_xml_charseq(head[0])
                em = EntityMention(
                    mention_id,
                    IntPair(start, end),
                    text,
                    entity_type+'.'+entity_subtype,
                    entity,
                    mention_type
                )
                doc.add_entity_mention(em)
                entity.mentions.append(em)
            doc.add_entity(entity)

    @classmethod
    def process_times(cls, doc, document_node):
        """
        :type doc: nlplingo.text.text_theory.Document
        :type document_node: xml.etree.ElementTree.Element
        """
        for time_node in document_node.findall('timex2'):
            time_id = time_node.attrib['ID']

            all_mentions = time_node.findall('timex2_mention')
            entity = Entity(time_id)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib['ID']
                (text, start, end) = cls.process_xml_charseq(mention_node[0][0])
                em = EntityMention(mention_id, IntPair(start, end), text, 'Time', entity, 'time')
                doc.add_entity_mention(em)
                entity.mentions.append(em)
            doc.add_entity(entity)

    @classmethod
    def process_values(cls, doc, document_node):
        """
        :type doc: nlplingo.text.text_theory.Document
        :type document_node: xml.etree.ElementTree.Element
        """
        for value_node in document_node.findall('value'):
            value_id = value_node.attrib['ID']
            value_type = value_node.attrib['TYPE']

            all_mentions = value_node.findall('value_mention')
            entity = Entity(value_id)
            for mention_node in all_mentions:
                mention_id = mention_node.attrib['ID']
                (text, start, end) = cls.process_xml_charseq(mention_node[0][0])
                em = EntityMention(mention_id, IntPair(start, end), text, value_type, entity, 'value')
                doc.add_entity_mention(em)
                entity.mentions.append(em)
            doc.add_entity(entity)
