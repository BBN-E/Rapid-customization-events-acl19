
from nlplingo.text.text_span import Span
from nlplingo.text.text_span import Sentence

def find_srls_involving_span(span, sentence, target_roles=None):
    """Find the SRL mentions within sentence, that involves the given span (e.g. this could be head of entity mention)
    We can optionally restrict to certain SRL roles

    :type span: nlplingo.text.text_span.Span
    :type sentence: nlplingo.text.text_span.Sentence
    :type target_role: set(str)
    Returns: list[nlplingo.text.text_theory.SRL]
    """
    ret = []

    for srl in sentence.srls:
        role = find_srl_role_of_span(span, srl)
        if role is not None:
            # this means that the span is involved as one of the arguments of the current srl
            if target_roles is None or (role in target_roles):
                ret.append(srl)
    return ret


def find_srl_role_of_span(span, srl):
    """If span is involved as one of the arguments of the given srl, return the srl role. Else return None
    The SRL argument must completely cover the given span. Since SRL arguments do not overlap, there is only max
    1 SRL role to return

    :type span: nlplingo.text.text_span.Span
    :type srl: nlplingo.text.text_theory.SRL
    Returns: str
    """
    start = span.start_char_offset()
    end = span.end_char_offset()

    for role in srl.roles:
        for arg in srl.roles[role]:
            arg_start = arg.start_char_offset()
            arg_end = arg.end_char_offset()
            if arg_start <= start and end <= arg_end:
                return role
    return None
