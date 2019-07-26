from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import xml.etree.ElementTree as etree

from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import Event

from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import EventArgument
from nlplingo.common import utils
from nlplingo.common.utils import IntPair

NAMESPACES = {'dc' : 'http://schemas.datacontract.org/2004/07/Enote.Model'}


class EnoteArgument(object):
    def __init__(self, name, text, start, end):
        self.name = name
        self.text = text
        self.start = start
        self.end = end

    @classmethod
    def from_xml_node(cls, node):
        name = node.find('dc:Name', NAMESPACES).text
        text = node.find('dc:Text', NAMESPACES).text
        start = int(node.find('dc:Start', NAMESPACES).text)
        end = int(node.find('dc:End', NAMESPACES).text)

        if start==0 and end==0:
            return None

        try:
            utext = text.decode('UTF8')
        except UnicodeEncodeError:
            utext = text

        num_of_right_spaces = len(utext) - len(utext.rstrip())
        if num_of_right_spaces > 0:
            utext = utext[:-num_of_right_spaces]
            end = end - num_of_right_spaces

        return EnoteArgument(name, utext, start, end)


def process_enote_file(doc, xml_file, auto_adjust):
    """Parses ENote annotation file to annotated_document.DocumentAnnotation

    :param all_text: raw text corresponding to the xml_file
    :param xml_file: ENote annotation file
    :param docid: string representing docid
    :param auto_adjust: Adjust annotation (start, end) position to match text. Useful if annotation data is noisy.
    :return: document_annotation.DocumentAnnotation
    """
    tree = etree.parse(xml_file)
    root_node = tree.getroot()

    all_text = doc.text

    events_node = root_node.find('dc:Events', NAMESPACES)
    for event_index, event_node in enumerate(events_node):
        event_type = event_node.find('dc:Name', NAMESPACES).text.decode('UTF8')
        event_id = '{}-e{}'.format(doc.docid, event_index)
        event = Event(event_id, event_type)

        candidate_anchors = []
        candidate_arguments = []
        for argument_index, argument_node in enumerate(event_node.find('dc:Arguments', NAMESPACES)):
            argument = EnoteArgument.from_xml_node(argument_node)

            # Skip argument is empty
            if argument == None:
                continue

            start = argument.start
            end = argument.end

            unicode_text = all_text[start:end]
            if all_text[start:end] != argument.text and auto_adjust:
                start, end = utils.find_best_location(all_text, argument.text, start, end)
                unicode_text = all_text[start:end]

            # TODO : we could also treat the following as anchors:
            # - event_type == 'Vulnerability' and argument.name == 'Name'
            # - event_type == 'Exploit' and argument.name == 'Name'
            if argument.name == 'Anchor':
                anchor_id = '{}-t{}'.format(event_id, len(candidate_anchors))
                anchor = Anchor(anchor_id, IntPair(start, end), unicode_text, event_type)
                candidate_anchors.append(anchor)
                #if event.overlaps_with_anchor(anchor):
                #    print('Dropping overlapping anchor, %s' % (anchor.to_string()))
                #else:
                #    event.add_anchor(anchor)
            else:
                arg_id = '{}-a{}'.format(event_id, len(candidate_arguments))

                # get the entity mention associated with the event argument
                em = doc.get_entity_mention_with_span(start, end)
                if em is None:
                    print(
                        'Dropping event argument, as I cannot find an entity mention with same offsets. %s (%d,%d) "%s" %s' % (
                        doc.docid, start, end, unicode_text.encode('ascii','ignore'), argument.name.decode('UTF8')))
                else:
                    arg = EventArgument(arg_id, em, argument.name.decode('UTF8'))
                    candidate_arguments.append(arg)
                    #event.add_argument(arg)

        for anchor in candidate_anchors:
            if event.overlaps_with_anchor(anchor):
                print('Dropping overlapping anchor, %s' % (anchor.to_string()))
            else:
                event.add_anchor(anchor)
        for arg in candidate_arguments:
            if event.overlaps_with_anchor(arg):
                print('Dropping argument that overlaps with anchor, %s' % (arg.to_string()))
            else:
                event.add_argument(arg)

        doc.add_event(event)

    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enote')
    parser.add_argument('--text')
    parser.add_argument('--docid')

    args = parser.parse_args()

    f = codecs.open(args.text, 'r', encoding='utf-8')
    all_text = f.read()
    f.close()

    doc = Document(args.docid, all_text.strip())
    doc = process_enote_file(doc, args.enote, auto_adjust=True)
    print(doc.to_string())
