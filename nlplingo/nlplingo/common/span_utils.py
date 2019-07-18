import math
import sys

from nlplingo.common.utils import IntPair


def offset_overlap(offset1, offset2):
    """
    :type offset1: nlplingo.common.utils.IntPair
    :type offset2: nlplingo.common.utils.IntPair
    """
    if offset1.first <= offset2.first and offset2.first < offset1.second:
        return True
    elif offset2.first <= offset1.first and offset1.first < offset2.second:
        return True
    else:
        return False

def offset_same(offset1, offset2):
    """
    :type offset1: nlplingo.common.utils.IntPair
    :type offset2: nlplingo.common.utils.IntPair
    """
    if offset1.first == offset2.first and offset1.second == offset2.second:
        return True
    else:
        return False

def find_best_location(all_text, search_string, start_idx, end_idx):
    """When all_text[start_idx, end_idx] != search_string, we use this method to find the location of
    search_string within all_text, that is closest to the given (start_idx, end_idx)."""

    search_string = search_string.strip()
    best_match = (len(all_text), None)
    start = 0

    while True:
        match_pos = all_text.find(search_string, start)
        if match_pos < 0:
            break
        dist = abs(start_idx - match_pos)
        if dist < best_match[0]:
            best_match = (dist, match_pos)
        start = match_pos + 1
    if best_match[1] is not None:
        #if config.verbose:
        #    print(u' Search string and indices mismatch: ' +
        #          u'"{0}" != "{1}". Found match by shifting {2} chars'.format(
        #              search_string, all_text[start_idx:end_idx],
        #              start_idx - best_match[1]))
        start_idx = best_match[1]
        end_idx = best_match[1] + len(search_string)
    else:
        raise Exception(u'Search string ({0}) not in text.'.format(search_string))

    return (start_idx, end_idx)

def get_span_with_offsets(spans, start, end):
    """Return the span with the exact same (start, end) offsets. There should only be one, or None
    :type spans: list[nlplingo.text.text_span.Span]
    Returns:
        text.text_span.Span
    """
    # exact_match = None
    # partial_matches = []
    # for span in spans:
    #     if start == span.start_char_offset() and end == span.end_char_offset():
    #         exact_match = span
    #     if start <= span.start_char_offset() and span.end_char_offset() <= end:
    #         partial_matches.append(span)
    #
    # if exact_match is not None:
    #     return exact_match
    # elif len(partial_matches) > 0:
    #     # return the one that has the largest end-char-offset
    #     rightmost_span = partial_matches[0]
    #     i = 1
    #     while i < len(partial_matches):
    #         if partial_matches[i].end_char_offset() > rightmost_span.end_char_offset():
    #             rightmost_span = partial_matches[i]
    #         i += 1
    #     return rightmost_span
    # else:
    #     return None
    for span in spans:
        if start == span.start_char_offset() and end == span.end_char_offset():
            return span
    return None

def get_spans_in_offsets(spans, start, end):
    """Return the list of spans (nlplingo.text.text_span.Span) that are within (start, end) offsets
    :type spans: list[nlplingo.text.text_span.Span]
    Returns:
        list[text.text_span.Span]
    """
    ret = []
    for span in spans:
        if start <= span.start_char_offset() and span.end_char_offset() <= end:
            ret.append(span)
    return ret

def get_tokens_covering_span(tokens, span):
    start_index = -1
    end_index = -1
    for i, token in enumerate(tokens):
        if token.start_char_offset() <= span.start_char_offset() and span.start_char_offset() <= token.end_char_offset():
            start_index = i
        if token.start_char_offset() <= span.end_char_offset() and span.end_char_offset() <= token.end_char_offset():
            end_index = i
    if start_index != -1 and end_index != -1:
        return tokens[start_index:end_index+1]
    else:
        return None

def get_tokens_corresponding_to_span(tokens, span):
    """Find the list of token(s) corresponding to span (of type text.text_span.Span)
    We look for exact offset match. If we cannot find an exact match, we return None
    :type tokens: list[nlplingo.text.text_span.Token]
    :type span: nlplingo.text.text_span.Span
    Returns:
        list[nlplingo.text.text_span.Token]
    """
    first_token_index = -1
    last_token_index = -1
    for i, token in enumerate(tokens):
        if span.start_char_offset() == token.start_char_offset():
            first_token_index = i
        if span.end_char_offset() == token.end_char_offset():
            last_token_index = i

    if first_token_index != -1 and last_token_index != -1:
        return tokens[first_token_index:last_token_index+1]
    else:
        return None

    # if first_token_index != -1 and last_token_index != -1:
    #     return tokens[first_token_index:last_token_index+1]
    # else:
    #     first_token_index = -1
    #     last_token_index = -1
    #     max_diff = sys.maxsize
    #     matching_first_token_index = -1
    #     span_prefix = span.text.split()[0]
    #     for i, token in enumerate(tokens):
    #         if token.text == span_prefix:
    #             diff = int(math.fabs(span.start_char_offset() - token.start_char_offset()))
    #             if diff < max_diff:
    #                 max_diff = diff
    #                 matching_first_token_index = i
    #
    #     if matching_first_token_index != -1:
    #         diff = span.start_char_offset() - tokens[matching_first_token_index].start_char_offset()
    #         span_start = span.start_char_offset() - diff
    #         span_end = span.end_char_offset() - diff
    #         for i, token in enumerate(tokens):
    #             if span_start == token.start_char_offset():
    #                 first_token_index = i
    #             if span_end == token.end_char_offset():
    #                 last_token_index = i
    #
    #         if first_token_index != -1 and last_token_index != -1:
    #             print('In get_tokens_corresponding_to_span: target="{}":{}-{}, approximate-matching="{}":{}-{}'.format(
    #                 span.text, span.start_char_offset(), span.end_char_offset(),
    #                 ' '.join(t.text for t in tokens[first_token_index:last_token_index + 1]), span_start, span_end))
    #             return tokens[first_token_index:last_token_index + 1]
    #
    # return None

