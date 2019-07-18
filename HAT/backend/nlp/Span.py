from nlp.common import Marking
class Span(object):
    def __init__(self,start_offset,end_offset):
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __eq__(self, other):
        if not isinstance(other,type(self)):
            return False
        return self.start_offset == other.start_offset and self.end_offset == other.end_offset

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (self.__class__.__name__,self.start_offset,self.end_offset).__hash__()

    def __str__(self):
        return (self.start_offset,self.end_offset).__str__()

    def __repr__(self):
        return self.__str__()

class CharOffsetSpan(Span):
    pass

class TokenIdxSpan(Span):
    pass

