from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import argparse
import codecs
import re

from nlplingo.text.text_theory import Document
from nlplingo.text.text_theory import Event
from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import EventSpan
from nlplingo.text.text_span import EventArgument
from nlplingo.text.text_span import EntityMention
from nlplingo.common.utils import IntPair

def read_event_annotation(filename, doc):
    """Reads event annotation from filename, and add to doc

    :type filename: str
    :type doc: cyberlingo.text.text_theory.Document

    <Event type="CloseAccount">
    CloseAccount	0	230
    anchor	181	187
    CloseAccount/Source	165	170
    CloseAccount/Source	171	175
    CloseAccount/Source	176	180
    CloseAccount/Target	191	198
    CloseAccount/Target	207	214
    CloseAccount/Target	215	229
    </Event>
    """
    lines = []
    """:type: list[str]"""
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('<Event type='):
            event_type = re.search(r' type="(.*?)"', line).group(1)
            event_id = '{}.e-{}'.format(doc.docid, len(doc.events))
            event = Event(event_id, event_type)

            i += 1
            line = lines[i]
            while not line.startswith('</Event>'):
                tokens = line.split()
                info = tokens[0]
                offset = IntPair(int(tokens[1]), int(tokens[2]))
                text = doc.text[offset.first:offset.second]

                if info == event_type: # this is an event span
                    id = '{}.s-{}'.format(event_id, len(event.event_spans))
                    event.add_event_span(EventSpan(id, offset, text, event_type))
                elif info == 'anchor': # anchor span
                    id = '{}.t-{}'.format(event_id, len(event.anchors))
                    event.add_anchor(Anchor(id, offset, text, event_type))
                elif '/' in info:      # argument span
                    em = EntityMention('dummy', offset, text, 'dummy')
                    arg_role = info[info.index('/')+1:]
                    id = '{}.t-{}'.format(event_id, len(event.arguments))
                    event.add_argument(EventArgument(id, em, arg_role))
                i += 1
                line = lines[i]
            doc.add_event(event)
        i += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')    # dir containing text files and annotations
    args = parser.parse_args()

    document = dict()
    for filename in glob.glob(args.datadir+'/*.txt'):
        docid = os.path.basename(filename)[:-4]
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            document[docid] = Document(docid, text=f.read().strip())

    for filename in glob.glob(args.datadir + '/*.meta'):
        docid = os.path.basename(filename)[:-5]
        read_event_annotation(filename, document[docid])

    for docid, doc in document.items():
        print('==== {} ===='.format(docid))
        print(doc.to_string())
        for event in doc.events:
            print(event.to_string())
        print('')

