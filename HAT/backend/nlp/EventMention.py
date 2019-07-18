
import collections
from nlp.Span import TokenIdxSpan,CharOffsetSpan



class EventMentionInstanceIdentifierTokenIdxBase(object):
    def __init__(self,doc_id,sentence_id,trigger_idx_span):
        self.doc_id = doc_id
        self.sentence_id = sentence_id
        assert isinstance(trigger_idx_span,TokenIdxSpan)
        self.trigger_idx_span = trigger_idx_span


    def __eq__(self, other):
        if not isinstance(other, EventMentionInstanceIdentifierTokenIdxBase):
            return False
        return self.doc_id == other.doc_id and \
               self.sentence_id == other.sentence_id and \
               self.trigger_idx_span == other.trigger_idx_span

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (self.doc_id,self.sentence_id,self.trigger_idx_span).__hash__()

    def reprJSON(self):
        return {
                    'docId':self.doc_id,
                    'sentenceId':self.sentence_id,
                    'triggerSentenceTokenizedPosition':self.trigger_idx_span.start_offset,
                    'triggerSentenceTokenizedEndPosition':self.trigger_idx_span.end_offset
                }

    def __repr__(self):
        return self.reprJSON().__repr__()

    def __str__(self):
        return self.reprJSON().__str__()

    @staticmethod
    def fromJSON(instance_identifier_token_idx_base_dict):
        doc_id = instance_identifier_token_idx_base_dict['docId']
        sentence_id = instance_identifier_token_idx_base_dict['sentenceId']
        trigger_tokenized_start_idx = instance_identifier_token_idx_base_dict['triggerSentenceTokenizedPosition']
        trigger_tokenized_end_idx = instance_identifier_token_idx_base_dict['triggerSentenceTokenizedEndPosition']
        return EventMentionInstanceIdentifierTokenIdxBase(doc_id, sentence_id, TokenIdxSpan(trigger_tokenized_start_idx, trigger_tokenized_end_idx))

class EventMentionInstanceIdentifierCharOffsetBase(object):
    def __init__(self,doc_id,sentence_char_off_span,trigger_char_off_span):
        self.doc_id = doc_id
        assert isinstance(sentence_char_off_span, CharOffsetSpan)
        self.sentence_char_off_span = sentence_char_off_span
        assert isinstance(trigger_char_off_span,CharOffsetSpan)
        self.trigger_char_off_span = trigger_char_off_span

    def __eq__(self, other):
        if not isinstance(other, EventMentionInstanceIdentifierCharOffsetBase):
            return False
        return self.doc_id == other.doc_id and \
               self.sentence_char_off_span == other.sentence_char_off_span and \
               self.trigger_char_off_span == other.trigger_char_off_span

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (self.doc_id,self.sentence_char_off_span,self.trigger_char_off_span).__hash__()

    def __str__(self):
        return (self.doc_id,self.sentence_char_off_span,self.trigger_char_off_span).__str__()

    def __repr__(self):
        return self.__str__()


class EventArgumentMentionIdentifierTokenIdxBase(object):
    def __init__(self,event_mention_instance_identifier_token_idx_base,argument_idx_span):
        assert isinstance(event_mention_instance_identifier_token_idx_base,EventMentionInstanceIdentifierTokenIdxBase)
        self.event_mention_instance_identifier_token_idx_base = event_mention_instance_identifier_token_idx_base
        assert isinstance(argument_idx_span,TokenIdxSpan)
        self.argument_idx_span =argument_idx_span

    def __eq__(self, other):
        if not isinstance(other,EventArgumentMentionIdentifierTokenIdxBase):
            return False
        return self.event_mention_instance_identifier_token_idx_base == other.event_mention_instance_identifier_token_idx_base and self.argument_idx_span == other.argument_idx_span
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return (self.event_mention_instance_identifier_token_idx_base.__hash__(),self.argument_idx_span.__hash__()).__hash__()
    def __repr__(self):
        return self.reprJSON().__repr__()
    def __str__(self):
        return self.reprJSON().__str__()
    def reprJSON(self):
        ret = dict()
        ret.update(self.event_mention_instance_identifier_token_idx_base.reprJSON())
        ret['argumentSentenceTokenizedPosition'] = self.argument_idx_span.start_offset
        ret['argumentSentenceTokenizedEndPosition'] = self.argument_idx_span.end_offset
        return ret

    @staticmethod
    def fromJSON(argument_instance_identifier_token_idx_base_dict):
        doc_id = argument_instance_identifier_token_idx_base_dict['docId']
        sentence_id = argument_instance_identifier_token_idx_base_dict['sentenceId']
        trigger_tokenized_start_idx = argument_instance_identifier_token_idx_base_dict['triggerSentenceTokenizedPosition']
        trigger_tokenized_end_idx = argument_instance_identifier_token_idx_base_dict['triggerSentenceTokenizedEndPosition']
        argument_tokenized_start_idx = argument_instance_identifier_token_idx_base_dict['argumentSentenceTokenizedPosition']
        argument_tokenized_end_idx = argument_instance_identifier_token_idx_base_dict['argumentSentenceTokenizedEndPosition']
        event_mention_id = EventMentionInstanceIdentifierTokenIdxBase(doc_id, sentence_id, TokenIdxSpan(trigger_tokenized_start_idx, trigger_tokenized_end_idx))
        argument_span = TokenIdxSpan(argument_tokenized_start_idx,argument_tokenized_end_idx)
        return EventArgumentMentionIdentifierTokenIdxBase(event_mention_id,argument_span)
