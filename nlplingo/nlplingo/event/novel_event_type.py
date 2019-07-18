
import sys
import codecs

from nlplingo.common.io_utils import read_file_to_set
from nlplingo.common.utils import IntPair
from nlplingo.common.parameters import Parameters

from collections import defaultdict

class AnchorAnnotation(object):
    def __init__(self, docid, id, event_type, span_text, head_text, offset):
        """
        :type docid: str
        :type id: str
        :type event_type: str
        :type span_text: str
        :type head_text: str
        :type offset: IntPair
        """
        self.docid = docid
        self.id = id
        self.event_type = event_type
        self.span_text = span_text
        self.head_text = head_text
        self.offset = offset


class NovelEventType(object):
    def __init__(self, params):
        """
        :type params: nlplingo.common.parameters.Parameters
        """
        self.existing_types = read_file_to_set(params.get_string('existing_types'))
        self.new_types = read_file_to_set(params.get_string('new_types'))           # novel event types
        self.anchor_annotation = self._read_anchor_annotation(params.get_string('new_types_annotation'))
        """:type: dict[str, list[AnchorAnnotation]]"""


    def _read_anchor_annotation(self, filepath):

        ret = defaultdict(list)
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split('\t')
                docid = tokens[0]
                anchor_id = tokens[1]
                event_type = tokens[2]
                anchor_span_text = tokens[3]
                anchor_head_text = tokens[4]
                anchor_offset = IntPair(int(tokens[5]), int(tokens[6]))

                annotation = AnchorAnnotation(docid, anchor_id, event_type, anchor_span_text, anchor_head_text, anchor_offset)
                ret[docid].append(annotation)
        return ret

    def filter_train(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        ret = []
        for eg in examples:
            if eg.event_type in self.existing_types:
                ret.append(eg)
            elif eg.event_type in self.new_types:
                for annotation in self.anchor_annotation[eg.sentence.docid]:
                    if annotation.event_type == eg.event_type and \
                       annotation.offset.first == eg.token.start_char_offset() and annotation.offset.second == eg.token.end_char_offset():
                        ret.append(eg)
                        break
        return ret


    def filter_test(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        ret = []
        for eg in examples:
            if eg.event_type not in self.new_types:
                ret.append(eg)
        return ret

