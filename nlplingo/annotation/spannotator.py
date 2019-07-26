from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import argparse
import codecs
import re

from collections import defaultdict

from nlplingo.common import utils
from nlplingo.common.utils import IntPair
from nlplingo.common.span_utils import offset_overlap
from nlplingo.common.span_utils import offset_same
from nlplingo.text.text_span import EventSpan
from nlplingo.text.text_theory import Event
from nlplingo.text.text_span import EventArgument
from nlplingo.text.text_span import EntityMention
from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import TextSpan


class SpannotatorAnnotation(object):

    def __init__(self, annotation_dir, text_dir, output_dir):
        """
        :param annotation_dir: dir containing the various tasks, e.g. spannotator/db/tasks
        :param text_dir: a list of raw text files that were annotated
        """
        self.annotation_dir = annotation_dir
        self.text_dir = text_dir
        self.output_dir = output_dir

        self.positive_span_count = 0
        self.negative_span_count = 0
        self.discard_span_count = 0

        # dictionary of event_type to annotation dir, e.g. CloseAccount => ../data/spannotator/db/tasks/CloseAccount-GW5-dan
        self.event_type_annotation_dir = self.get_event_type_to_annotation_dir()
        self.target_filenames = self.aggregate_target_filenames()

    def get_event_type_to_annotation_dir(self):
        """
        Returns:
            dict[str, str]
        """
        ret = dict()
        for f in os.listdir(self.annotation_dir):
            filepath = os.path.join(self.annotation_dir, f)
            if os.path.isdir(filepath):
                event_type = f.split('-')[0]
                ret[event_type] = os.path.join(filepath, 'annotation')
        return ret

    def aggregate_target_filenames(self):
        ret = set()
        """:type: set[str]"""
        filecount = defaultdict(int)
        for event_type, annotation_dir in self.event_type_annotation_dir.items():
            for filename in os.listdir(annotation_dir):
                filecount[filename] += 1
        for filename, count in filecount.items():
            if count == 1:
                ret.add(filename)
        print('Kept {} out of {} files'.format(len(ret), len(filecount.keys())))
        return ret

    def read_spans(self, annotated_events):
        """From the annotation files, we capture the positive and negative spans,
        Then return a dictionary from filename or docid to list[text.text_span.TextSpan]
        The list orders the TextSpan by their start_char_offset. The text within each TextSpan is also normalized,
        with newlines replaced by space and consecutive spaces replaced by a single space.

        :type annotated_events: dict[str, list[nlplingo.text.text_theory.Event]]
        """
        ret = defaultdict(list)
        """:type: dict[str, list[nlplingo.text.text_span.TextSpan]]"""

        # We first collect the positive and negative spans, from the annotation files.
        # Note that the same file can be annotated multiple times (via different event types).
        # Need to de-dupliciate the spans later.
        file_spans = defaultdict(list)  # filename -> list[text.text_span.TextSpan]
        """:type: dict[str, list[nlplingo.text.text_span.TextSpan]]"""
        for event_type, annotation_dir in self.event_type_annotation_dir.items():
            for filename in os.listdir(annotation_dir):
                if filename not in self.target_filenames:
                    continue

                annotation_file = os.path.join(annotation_dir, filename)

                text_file = os.path.join(self.text_dir, filename)
                with codecs.open(text_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()

                spans = self._read_spans_from_file(annotation_file, event_type, raw_text, events=annotated_events[filename])
                file_spans[filename].extend(spans)

        # for each file, de-duplicate the spans and order them by their start-char-offset
        for filename in file_spans.keys():
            all_spans = file_spans[filename]
            """:type: list[nlplingo.text.text_span.TextSpan]"""

            current_spans = dict()  # start_char_offset -> TextSpan ; holds de-duplicated spans keyed by start-offset
            """:type: dict[int, nlplingo.text.text_span.TextSpan]"""
            for span in all_spans:
                # check whether 'span' is already in current_spans
                span_offset = IntPair(span.start_char_offset(), span.end_char_offset())
                to_add = True
                for start, s in current_spans.items():
                    s_offset = IntPair(s.start_char_offset(), s.end_char_offset())
                    if offset_same(span_offset, s_offset):
                        print('Found offset_same spans')
                        to_add = False
                        break
                    elif offset_overlap(span_offset, s_offset):
                        # we will remove both spans, just to reduce noise
                        print('Found offset_overlap spans in file {}, {}:{}'.format(filename, span_offset.to_string(), s_offset.to_string()))
                        print('[{}]\n==== vs ====\n[{}]\n'.format(span.text, s.text))
                        del current_spans[start]
                        to_add = False
                        break
                if to_add:
                    current_spans[span.start_char_offset()] = span

            if len(current_spans) > 0:
                for start_char_offset in sorted(current_spans):
                    span = current_spans[start_char_offset]
                    """:type: nlplingo.text.text_span.TextSpan"""
                    ret[filename].append(span)

        return ret

    def _read_spans_from_file(self, infile, event_type, text, events=None):
        """Get the positive and negative spans
        Returns:
            list[nlplingo.text.text_span.TextSpan]
        """
        ret = []
        with open(infile, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                span_type = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2]) + 1
                text_string = ' '.join(text[start:end].replace('\n', ' ').strip().split())
                end = start + len(text_string)

                if '<' in text_string or '>' in text_string:
                    print('Skipping annotation of type {}, as it has either "<" or ">"'.format(span_type))
                    continue

                span_offset = IntPair(start, end)

                if span_type == event_type:
                    # if this is a positive span, then we need to make sure we have an event for it
                    if events is not None:
                        found_span = False
                        for event in events:
                            if offset_same(event.event_spans[0].int_pair, span_offset):
                                found_span = True
                                break
                        if found_span:
                            ret.append(TextSpan(span_offset, text_string))
                            self.positive_span_count += 1
                        else:
                            self.discard_span_count += 1
                    else:
                        self.discard_span_count += 1
                elif span_type == 'negative':
                    ret.append(TextSpan(span_offset, text_string))
                    self.negative_span_count += 1
        return ret


    @staticmethod
    def _find_event_containing_span(events, start, end):
        """
        :type events: list[nlplingo.text.text_theory.Event]
        :type start: int
        :type end: int
        """
        for event in events:    # in this context, each event is guaranteed to have a single EventSpan
            if event.event_spans[0].start_char_offset() <= start and end <= event.event_spans[0].end_char_offset():
                return event
        return None

    def read_annotation_files(self):
        """
        Returns:
            dict[str, list[nlplingo.text.text_theory.Event]]
        """
        ret = defaultdict(list)

        for event_type, annotation_dir in self.event_type_annotation_dir.items():
            for filename in os.listdir(annotation_dir):
                if filename not in self.target_filenames:
                    continue

                annotation_file = os.path.join(annotation_dir, filename)

                text_file = os.path.join(self.text_dir, filename)
                with codecs.open(text_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()

                events = self._read_annotation_file(annotation_file, event_type, raw_text)
                ret[filename].extend(events)
        return ret

    @classmethod
    def _read_annotation_file(cls, infile, event_type, text):
        """
        :type infile: str
        :type event_type: str
        :type text: str
        Returns:
            list[nlplingo.text.text_theory.Event]
        :param text: this is the raw text corresponding to the annotation
        """
        docid = os.path.basename(infile)

        events = []
        """:type: list[nlplingo.text.text_theory.Event]"""
        negative_spans = []
        """:type: list[nlplingo.text.text_span.TextSpan]"""
        anchors_not_in_eventspans = []      # these might be in negative spans
        """:type: list[nlplingo.text.text_span.Anchor]"""
        with open(infile, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                span_type = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2]) + 1
                text_string = ' '.join(text[start:end].replace('\n', ' ').strip().split())
                end = start + len(text_string)

                if '<' in text_string or '>' in text_string:
                    print('Skipping annotation of type {}, as it has either "<" or ">"'.format(span_type))
                    continue

                if span_type == event_type:
                    id = '{}-e{}'.format(docid, len(events))
                    event_span = EventSpan(id, IntPair(start, end), text_string, event_type)
                    e = Event(id, event_type)
                    e.add_event_span(event_span)
                    events.append(e)
                elif '/' in span_type:  # this is an event argument
                    em = EntityMention('dummy', IntPair(start, end), text_string, 'dummy')
                    event_role = span_type.split('/')[1]
                    e = cls._find_event_containing_span(events, start, end)
                    if e is None:
                        print('Cannot find an event span for {} {} (start,end)=({},{}) "{}". Skipping.'.format(event_type, docid, start, end, text_string))
                    else:
                        arg_id = '{}-a{}'.format(e.id, e.number_of_arguments())
                        e.add_argument(EventArgument(arg_id, em, event_role))
                elif span_type == 'anchor':
                    e = cls._find_event_containing_span(events, start, end)
                    anchor = Anchor('dummy', IntPair(start, end), text_string, event_type)
                    if e is None:
                        # it might be in a negative span
                        #print('Cannot find an event span for {} {} (start,end)=({},{}) "{}". Skipping.'.format(event_type, docid, start, end, text_string.replace(' ', '_')))
                        anchors_not_in_eventspans.append(anchor)
                    else:
                        e.add_anchor(anchor)
                elif span_type == 'negative':
                    negative_spans.append(TextSpan(IntPair(start, end), text_string))
                elif span_type == 'interesting':
                    pass                # we discard these for now

        for anchor in anchors_not_in_eventspans:
            found = False
            for span in negative_spans:
                if span.start_char_offset() <= anchor.start_char_offset() and anchor.end_char_offset() <= span.end_char_offset():
                    found = True
                    break
            if not found:
                print('Cannot find an event nor negative span for anchor {} {} (start,end)=({},{}) "{}". Skipping.'.format( \
                    event_type, docid, anchor.start_char_offset(), anchor.end_char_offset(), anchor.text.replace(' ', '_')))

        # keep only events with anchor
        return [event for event in events if event.number_of_anchors() > 0]

    @classmethod
    def adjust_and_write_annotation_offset(cls, file_spans, annotated_events, output_dir):
        """Since we keep only the positive and negative spans from the original text files,
        we need to adjust the annotation offsets accordingly.

        :type file_spans: dict[str, list[nlplingo.text.text_span.TextSpan]]
        :type annotated_events: dict[str, list[nlplingo.text.text_theory.Event]]

        The keys for both dictionaries are filenames. Note that the filename keys in annotated_events is a subset of
        the filename keys in file_spans, since some files might contain only negative spans.
        """

        for filename, events in annotated_events.items():
            outlines = []           # strings storing adjusted annotation offsets
            """:type: list[str]"""

            spans = file_spans[filename]
            """:type: list[nlplingo.text.text_span.TextSpan]"""

            # establish the new offsets for spans
            new_offsets = []
            """:type: list[nlplingo.common.utils.IntPair]"""
            offset = 0
            for span in spans:
                end = offset + len(span.text)
                new_offsets.append(IntPair(offset, end))
                offset = end + 1    # +1 for the newline

            for event in events:
                event_span = event.event_spans[0]
                # find the index of this event_span in 'spans'
                span_index = -1
                for i, span in enumerate(spans):
                    if offset_same(span.int_pair, event_span.int_pair):
                        span_index = i
                        break
                if span_index == -1:
                    raise ValueError('Could not find a corresponding span, should not happen')

                span_start = spans[span_index].start_char_offset()
                text = spans[span_index].text
                new_offset = new_offsets[span_index]

                outlines.append('<Event type="{}">'.format(event.label))
                outlines.append('{}\t{}\t{}'.format(event.label, new_offset.first, new_offset.second))

                if event.number_of_anchors() == 0:
                    raise ValueError('An event should have at least 1 anchor!')

                for anchor in event.anchors:
                    start = anchor.start_char_offset() - span_start
                    end = anchor.end_char_offset() - span_start
                    if text[start:end] != anchor.text:
                        new_start, new_end = utils.find_best_location(text, anchor.text, start, end)
                        print('Adjusting anchor offsets from ({},{}) to ({},{})'.format(start, end, new_start, new_end))
                        start = new_start
                        end = new_end
                    start += new_offset.first
                    end += new_offset.first
                    outlines.append('anchor\t{}\t{}'.format(start, end))

                for arg in event.arguments:
                    start = arg.start_char_offset() - span_start
                    end = arg.end_char_offset() - span_start
                    if text[start:end] != arg.text:
                        new_start, new_end = utils.find_best_location(text, arg.text, start, end)
                        print('Adjusting argument offsets from ({},{}) to ({},{})'.format(start, end, new_start, new_end))
                        start = new_start
                        end = new_end
                    start += new_offset.first
                    end += new_offset.first
                    outlines.append('{}/{}\t{}\t{}'.format(event.label, arg.label, start, end))

                outlines.append('</Event>')

            if len(outlines) > 0:
                with open(os.path.join(output_dir, filename+'.meta'), 'w') as f:
                    for line in outlines:
                        f.write(line + '\n')


    def write_spans(self, file_spans, output_dir):
        """
        :type file_spans: dict[str, list[nlplingo.text.text_span.TextSpan]]
        """
        for filename, spans in file_spans.items():
            if len(spans) > 0:
                with codecs.open(os.path.join(output_dir, filename+'.txt'), 'w', encoding='utf-8') as f:
                    for span in spans:
                        f.write(span.text)
                        f.write('\n')

    @staticmethod
    def print_event_annotations(doc):
        """
        :type doc: nlplingo.text.text_theory.Document
        """
        ret = []
        for sentence in doc.sentences:
            for event in sentence.events:
                et = event.label
                ret.append('<Event type="{}">'.format(et))
                ret.append('{}\t{}\t{}'.format(et, sentence.start_char_offset(), sentence.end_char_offset()))
                for anchor in event.anchors:
                    ret.append('anchor\t{}\t{}'.format(anchor.start_char_offset(), anchor.end_char_offset()))
                for arg in event.arguments:
                    ret.append('{}/{}\t{}\t{}'.format(et, arg.label, arg.start_char_offset(), arg.end_char_offset()))
                ret.append('</Event>')
        return ret


def print_stats_on_events(annotated_events, spannotator):
    """
    :type annotated_events: dict[str, list[nlplingo.text.text_theory.Event]]
    :type spannotator: SpannotatorAnnotation
    """
    event_count = 0
    event_with_anchor_count = 0
    event_with_source_arg_count = 0
    event_with_target_arg_count = 0
    for filename, file_events in annotated_events.items():
        for event in file_events:
            event_count += 1
            args = defaultdict(list)
            if event.number_of_anchors() > 0:
                event_with_anchor_count += 1
            else:
                print('Event {} without anchor "{}"'.format(event.label, event.event_spans[0].text))
            for arg in event.arguments:
                #print('arg\t{}\t[{}]\t{}'.format(arg.label, arg.text, event.event_spans[0].text))
                args[arg.label].append(arg)
            if 'Source' in args.keys():
                event_with_source_arg_count += 1
            if 'Target' in args.keys():
                event_with_target_arg_count += 1
    print('#events={}, #with_anchor={}, #with_Source_arg={}, #with_Target_arg={}'.format(event_count, event_with_anchor_count, event_with_source_arg_count, event_with_target_arg_count))
    print('#positive_spans={}, #negative_spans={}'.format(spannotator.positive_span_count, spannotator.negative_span_count))


def remove_trailing_periods(text, offset):
    """
    :type text: str
    :type offset: IntPair
    """
    newtext = text
    newoffset = IntPair(offset.first, offset.second)
    chars = set(['.', ',', ':', ';', ')', '}', ']', '"', '\'', '?', '!'])
    if text[-1] in chars:
        i = 1
        while text[-(i+1)] is ' ':
            i += 1
        newtext = text[0:-i]
        newoffset.second = newoffset.second - i
    return newtext, newoffset


def process_span_file(doc, filename):
    """Reads event annotation from filename, and add to doc

    :type filename: str
    :type doc: nlplingo.text.text_theory.Document

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

                if info == event_type or info == 'anchor' or '/' in info:
                    text = doc.get_text(offset.first, offset.second)
                    if text is None or text == '':
                        print('WARNING: skipping annotation span {} {}-{}'.format(doc.docid, offset.first, offset.second))
                    else:
                        # sometimes, the UI captures an extra trailing space. Check for that and adjust ending offset
                        if text[-1] == ' ':
                            text = text[0:-1]
                            offset.second = offset.second - 1

                        if info == event_type:  # this is an event span
                            id = '{}.s-{}'.format(event_id, len(event.event_spans))
                            event.add_event_span(EventSpan(id, offset, text, event_type))
                        elif info == 'anchor':  # anchor span
                            id = '{}.t-{}'.format(event_id, len(event.anchors))
                            #print('Spannotator, adding ANCHOR with text "{}"'.format(text))
                            newtext, newoffset = remove_trailing_periods(text, offset)
                            if text != newtext:
                                print('- revising anchor, text=[%s] offset=(%d,%d) newtext=[%s] newoffset=(%d,%d)' % (text, offset.first, offset.second, newtext, newoffset.first, newoffset.second))
                            event.add_anchor(Anchor(id, newoffset, newtext, event_type))
                        elif '/' in info:  # argument span
                            em_id = 'm-{}-{}'.format(offset.first, offset.second)
                            newtext, newoffset = remove_trailing_periods(text, offset)
                            if text != newtext:
                                print('- revising argument, text=[%s] offset=(%d,%d) newtext=[%s] newoffset=(%d,%d)' % (text, offset.first, offset.second, newtext, newoffset.first, newoffset.second))
                            em = EntityMention(em_id, newoffset, newtext, 'dummy')
                            # we just use a dummy em first, for creating the EventArgument (notice that this em is not added to the doc)
                            # later, when we annotate sentence, we will find an actual EntityMention that is backed by tokens
                            # and use that to back the EventArgument
                            # Ref: text_theory.annotate_sentence_with_events()
                            arg_role = info[info.index('/') + 1:]
                            arg_id = '{}.t-{}'.format(event_id, len(event.arguments))
                            event.add_argument(EventArgument(arg_id, em, arg_role))

                i += 1
                line = lines[i]
            doc.add_event(event)
        i += 1
    return doc


if __name__ == "__main__":
    """
    Example command: PYTHONPATH=. python annotation/spannotator.py --annotation_dir ../data/spannotator/db/tasks --text_dir ../data/spannotator/text --output_dir ../data/spannotator/output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_dir')
    parser.add_argument('--text_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    spannotator = SpannotatorAnnotation(args.annotation_dir, args.text_dir, args.output_dir)

    annotated_events = spannotator.read_annotation_files()  # a list of annotated Event for each filename
    """:type: dict[str, list[nlplingo.text.text_theory.Event]]"""

    file_spans = spannotator.read_spans(annotated_events)                   # a list of TextSpan for each filename
    """:type: dict[str, list[nlplingo.text.text_span.TextSpan]]"""

    print('\n==== Stats on Event annotation ====')
    print_stats_on_events(annotated_events, spannotator)

    spannotator.adjust_and_write_annotation_offset(file_spans, annotated_events, args.output_dir)

    spannotator.write_spans(file_spans, args.output_dir)


