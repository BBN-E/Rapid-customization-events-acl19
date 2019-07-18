import copy

from data_access_layer.corpus_management.mongodb import insert_or_update_sentence
from models.frontend import TriggerArgumentInstance
from nlp.common import Marking
import pymongo
from nlp.Span import CharOffsetSpan
from nlp.EventMention import EventMentionInstanceIdentifierCharOffsetBase,EventArgumentMentionIdentifierTokenIdxBase


class ArgumentAnnotationMongoDBAccessLayer(object):
    def get_paginate_argument_metadata(self,session_id,pagination_size,mongo_instance):
        cnt_dict = dict()
        for i in mongo_instance['{}_argument_marking'.format(session_id)].find({},{'_id':0,'eventType':1,'argumentType':1}):
            cnt_dict[i['eventType']][i['argumentType']] = cnt_dict.setdefault(i['eventType'],dict()).get(i['argumentType'],0) + 1
        ret = dict()
        for event_type,argument_cnt_dic in cnt_dict.items():
            for argument_type,cnt in argument_cnt_dic.items():
                ret.setdefault(event_type,dict())[argument_type] = cnt_dict[event_type][argument_type] // pagination_size
        return ret

    def get_paginate_argument_candidates(self,session_id,event_type_id,pagination_size,page,mongo_instance,doc_id_table_name,argument_type=None):
        query = {'eventType':event_type_id}
        if argument_type is None:
            query['argumentType'] = "trigger"
        else:
            query['argumentType'] = argument_type
        docid_to_corpus_cache = dict()

        buf = list()

        # unannotated instance first?

        for i in mongo_instance['{}_argument_marking'.format(session_id)].find(query).skip(
            page * pagination_size).limit(pagination_size):
            if i['docId'] in docid_to_corpus_cache:
                corpus_name = docid_to_corpus_cache[i['docId']]['corpusName']
            else:
                en = mongo_instance[doc_id_table_name].find_one({'docId':i['docId']})
                if en is None:
                    raise ValueError
                else:
                    docid_to_corpus_cache[i['docId']] = en
                    corpus_name = en['corpusName']
            sentence_info = \
                mongo_instance["{}_sentence_info".format(corpus_name)].find_one({'docId': i['docId'], 'sentenceId': i['sentenceId']})[
                'sentenceInfo']
            buf.append(TriggerArgumentInstance.fromMongoDB(i,sentence_info))

        # A ranking function here?

        return buf

    def dump_candidates(self,candidate_list,mongo_instance,session_id,doc_id_table_name):
        docid_to_corpus_cache = dict()
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("docId", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("sentenceId", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("triggerSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("triggerSentenceTokenizedEndPosition", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("eventType", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("argumentType", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("argumentSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)
        mongo_instance['{}_argument_marking'.format(session_id)].create_index(
            [("argumentSentenceTokenizedEndPosition", pymongo.ASCENDING)], background=True)
        for instance in candidate_list:
            if instance['docId'] not in docid_to_corpus_cache:
                en = mongo_instance[doc_id_table_name].find_one({'docId':instance['docId']})
                if en is None:
                    raise ValueError
                else:
                    docid_to_corpus_cache[instance['docId']] = en
            corpus_name = docid_to_corpus_cache[instance['docId']]['corpusName']
            insert_or_update_sentence(mongo_instance,"{}_sentence_info".format(corpus_name),instance["docId"],instance["sentenceId"],{"token":instance['token'],"tokenSpan":instance['tokenSpan']})
            event_argument_id = EventArgumentMentionIdentifierTokenIdxBase.fromJSON(instance)
            self.modify_annotation(mongo_instance,
                                   session_id,
                                   event_argument_id,
                                   instance['eventType'],
                                   instance['argumentType']
                                   , False, Marking.POSITIVE,False)

    def modify_annotation(self,mongo_instance,session,event_argument_id,event_type,argument_type,touched,marking,override=False):
        assert isinstance(event_argument_id,EventArgumentMentionIdentifierTokenIdxBase)
        query_key = event_argument_id.reprJSON()
        query_key['eventType'] = event_type
        query_key['argumentType'] = argument_type
        db_entry = mongo_instance['{}_argument_marking'.format(session)].find_one(query_key)
        if db_entry is not None:
            if override is True:
                db_entry['positive'] = marking
                db_entry['touched'] = touched
                mongo_instance['{}_argument_marking'.format(session)].find_one_and_replace(query_key, db_entry)
        else:
            en = copy.deepcopy(query_key)
            en['positive'] = marking
            en['touched'] = touched
            mongo_instance['{}_argument_marking'.format(session)].insert_one(en)

    def dump_annotation_out(self,mongo_instance,session,doc_id_table_name):
        sentence_info_cache = dict()
        doc_info_cache = dict()
        ret = dict()
        for instance in mongo_instance['{}_argument_marking'.format(session)].find({}):
            event_argument_id = EventArgumentMentionIdentifierTokenIdxBase.fromJSON(instance)
            event_type = instance['eventType']
            argument_type = instance['argumentType']

            if event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id not in doc_info_cache:
                en = mongo_instance[doc_id_table_name].find_one({'docId':event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id})
                if en is None:
                    raise ValueError
                doc_info_cache[event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id] = en
            doc_info = doc_info_cache[event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id]
            corpus_name = doc_info['corpusName']
            doc_path = doc_info['docPath']
            if (event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id,event_argument_id.event_mention_instance_identifier_token_idx_base.sentence_id) not in sentence_info_cache:
                en = mongo_instance["{}_sentence_info".format(corpus_name)].find_one({"docId":event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id,
                                                                                      "sentenceId":event_argument_id.event_mention_instance_identifier_token_idx_base.sentence_id
                                                                                      })
                if en is None:
                    raise ValueError
                sentence_info_cache[(event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id,event_argument_id.event_mention_instance_identifier_token_idx_base.sentence_id)] = en
            sentence_info = sentence_info_cache[(event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id,event_argument_id.event_mention_instance_identifier_token_idx_base.sentence_id)]['sentenceInfo']
            sentence_start_char = sentence_info['tokenSpan'][0]['key']
            sentence_end_char = sentence_info['tokenSpan'][-1]['value']
            trigger_start_char = sentence_info['tokenSpan'][event_argument_id.event_mention_instance_identifier_token_idx_base.trigger_idx_span.start_offset]['key']
            trigger_end_char = sentence_info['tokenSpan'][event_argument_id.event_mention_instance_identifier_token_idx_base.trigger_idx_span.end_offset]['value']
            argument_start_char = sentence_info['tokenSpan'][event_argument_id.argument_idx_span.start_offset]['key']
            argument_end_char = sentence_info['tokenSpan'][event_argument_id.argument_idx_span.end_offset]['value']
            event_mention_id = EventMentionInstanceIdentifierCharOffsetBase(
                event_argument_id.event_mention_instance_identifier_token_idx_base.doc_id,
                CharOffsetSpan(sentence_start_char,sentence_end_char),
                CharOffsetSpan(trigger_start_char,trigger_end_char)
            )
            trigger_word = " ".join(sentence_info['token'][event_argument_id.event_mention_instance_identifier_token_idx_base.trigger_idx_span.start_offset:event_argument_id.event_mention_instance_identifier_token_idx_base.trigger_idx_span.end_offset+1])
            ret.setdefault((doc_path,event_mention_id,event_type,trigger_word),set()).add((argument_type,argument_start_char,argument_end_char,instance['touched'],instance['positive']))
        return ret
