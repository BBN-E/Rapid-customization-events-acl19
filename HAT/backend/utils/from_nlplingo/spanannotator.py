
import codecs,re,os

from nlp.Span import CharOffsetSpan
from nlp.EventMention import EventMentionInstanceIdentifierCharOffsetBase,EventArgumentMentionIdentifierTokenIdxBase

def process_span_file(filename):
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
    event_mention_to_type_set = dict()
    event_mention_to_argument_set = dict()
    while i < len(lines):
        line = lines[i]
        if line.startswith('<Event type='):
            event_type = re.search(r' type="(.*?)"', line).group(1)
            i += 1
            line = lines[i]
            sent_char_off = None
            trigger_char_off = None
            argument_type_to_char_off = dict()
            while not line.startswith('</Event>'):
                tokens = line.split()
                info = tokens[0]
                offset = CharOffsetSpan(int(tokens[1]),int(tokens[2])-1)

                if info == event_type or info == 'anchor' or '/' in info:
                    if info == event_type:  # this is an event span
                        sent_char_off = offset
                    elif info == 'anchor':  # anchor span
                        trigger_char_off = offset
                    elif '/' in info:  # argument span
                        arg_role = info[info.index('/') + 1:]
                        argument_type_to_char_off.setdefault(arg_role,list()).append(offset)
                i += 1
                line = lines[i]
            if sent_char_off is None or trigger_char_off is None:
                print("WARNING: We're losing thing")
                continue
            else:
                event_mention = EventMentionInstanceIdentifierCharOffsetBase(os.path.basename(filename).split(".")[0],sent_char_off,trigger_char_off)
                event_mention_to_type_set.setdefault(event_mention,set()).add(event_type)
                l = event_mention_to_argument_set.setdefault(event_mention,dict())
                for arg_role,offsets in argument_type_to_char_off.items():
                    l.setdefault(arg_role,set()).update(offsets)
        i += 1
    return event_mention_to_type_set,event_mention_to_argument_set

