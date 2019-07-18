from nlp.EventMention import EventMentionInstanceIdentifierTokenIdxBase
from nlp.Span import TokenIdxSpan
from nlp.common import Marking

class TriggerArgumentInstance(object):
    def __init__(self,instance_identifier_token_idx_base,token_array,event_type,argument_type=None,argument_idx_span = None,marking =None,touched=False):
        assert isinstance(instance_identifier_token_idx_base, EventMentionInstanceIdentifierTokenIdxBase)
        self.instance_identifier_token_idx_base = instance_identifier_token_idx_base
        self.token_array = token_array
        self.event_type = event_type
        self.argument_type = "trigger" if argument_type is None else argument_type
        if argument_idx_span is None:
            self.argument_idx_span = self.instance_identifier_token_idx_base.trigger_idx_span
        else:
            assert isinstance(argument_idx_span,TokenIdxSpan)
            self.argument_idx_span = argument_idx_span


        if marking is None:
            self.marking = True
        else:
            self.marking = marking

        self.touched = touched

    def reprJSON(self):
        tags = dict()
        tags['trigger'] = [j for j in range(self.instance_identifier_token_idx_base.trigger_idx_span.start_offset,self.instance_identifier_token_idx_base.trigger_idx_span.end_offset+1)]
        tags[self.argument_type] = [j for j in range(self.argument_idx_span.start_offset,self.argument_idx_span.end_offset+1)]

        if tags['trigger'][0] - 2 < 0:
            abstract = self.token_array[0:5]
        elif tags['trigger'][-1] + 2 >= len(self.token_array):
            abstract = self.token_array[-5:]
        else:
            abstract = self.token_array[tags['trigger'][0] - 2:tags['trigger'][-1] + 3]
        abstract = " ".join(abstract)

        return {"key":"{}.{}".format(self.instance_identifier_token_idx_base.doc_id,self.instance_identifier_token_idx_base.sentence_id),
                            "type":"sentence",
                            "aux":{
                                "instanceId": self.instance_identifier_token_idx_base.reprJSON(),
                                "token_array":self.token_array,
                                "abstract":abstract,
                                "tags":tags,
                                "marking":self.marking,
                                "touched":self.touched,
                                "eventType":self.event_type,
                                "argumentType":self.argument_type
                            }
                        }

    def __repr__(self):
        return self.reprJSON().__repr__()

    def __str__(self):
        return self.reprJSON().__str__()

    @staticmethod
    def fromJSON(instance_dict):
        instance_identifier_token_idx_base = EventMentionInstanceIdentifierTokenIdxBase.fromJSON(instance_dict['aux']["instanceId"])
        token_array = instance_dict['aux']['token_array']
        event_type = instance_dict['aux']['eventType']
        argument_type = instance_dict['aux'].get('argumentType',None)
        touched = instance_dict['aux'].get('touched',False)
        marking = instance_dict['aux'].get('marking',None)
        argument_start_idx = min(instance_dict['aux']['tags'][argument_type])
        argument_end_idx = max(instance_dict['aux']['tags'][argument_type])
        return TriggerArgumentInstance(instance_identifier_token_idx_base,token_array,event_type,argument_type,TokenIdxSpan(argument_start_idx,argument_end_idx),marking,touched)

    @staticmethod
    def fromMongoDB(marked_rec,sentence_info):
        instance_identifier_token_idx_base = EventMentionInstanceIdentifierTokenIdxBase.fromJSON(marked_rec)
        token_array = sentence_info['token']
        event_type = marked_rec.get('eventType',None)
        if marked_rec.get('argumentType',"trigger") == "trigger":
            argument_type = None
            argument_idx_span = None
            marking = marked_rec.get('positive',Marking.NEUTRAL)
        else:
            argument_type = marked_rec.get('argumentType',"trigger")
            argument_start_idx = marked_rec.get("argumentSentenceTokenizedPosition",-1)
            argument_end_idx = marked_rec.get('argumentSentenceTokenizedEndPosition',-1)
            argument_idx_span = TokenIdxSpan(argument_start_idx,argument_end_idx)
            marking = marked_rec.get('positive',Marking.NEUTRAL)
        touched = marked_rec.get('touched',False)
        return TriggerArgumentInstance(instance_identifier_token_idx_base, token_array, event_type, argument_type,
                                       argument_idx_span, marking,touched)