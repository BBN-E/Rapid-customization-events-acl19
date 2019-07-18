import copy,json

from nlp.Span import CharOffsetSpan
from nlp.EventMention import EventMentionInstanceIdentifierCharOffsetBase,EventMentionInstanceIdentifierTokenIdxBase
from models.frontend import TriggerArgumentInstance
from nlp.common import Marking

class TriggerAnnotationMongoDBAccessLayer(object):

    def change_event_type(self,mongo_instance,session,old_event_type,new_event_type):
        for instance in mongo_instance['{}_trigger_marking'.format(session)].find({'eventType':old_event_type}):
            instance['eventType'] = new_event_type
            mongo_instance['{}_trigger_marking'.format(session)].find_one_and_replace({'_id':instance['_id']},instance)

    def dump_annotation_out(self,mongo_instance,session,doc_id_table_name):

        sentence_info_cache = dict()
        doc_info_cache = dict()
        ret = dict()
        for instance in mongo_instance['{}_trigger_marking'.format(session)].find({}):
            event_id = EventMentionInstanceIdentifierTokenIdxBase.fromJSON(instance)
            event_type = instance['eventType']
            if event_type == "dummy":
                continue
            if event_id.doc_id not in doc_info_cache:
                en = mongo_instance[doc_id_table_name].find_one({'docId':event_id.doc_id})
                if en is None:
                    raise ValueError
                doc_info_cache[event_id.doc_id] = en
            doc_info = doc_info_cache[event_id.doc_id]
            corpus_name = doc_info['corpusName']
            doc_path = doc_info['docPath']
            if (event_id.doc_id,event_id.sentence_id) not in sentence_info_cache:
                en = mongo_instance["{}_sentence_info".format(corpus_name)].find_one({"docId":event_id.doc_id,
                                                                                      "sentenceId":event_id.sentence_id
                                                                                      })
                if en is None:
                    raise ValueError
                sentence_info_cache[(event_id.doc_id,event_id.sentence_id)] = en
            sentence_info = sentence_info_cache[(event_id.doc_id,event_id.sentence_id)]['sentenceInfo']
            sentence_start_char = sentence_info['tokenSpan'][0]['key']
            sentence_end_char = sentence_info['tokenSpan'][-1]['value']
            trigger_start_char = sentence_info['tokenSpan'][event_id.trigger_idx_span.start_offset]['key']
            trigger_end_char = sentence_info['tokenSpan'][event_id.trigger_idx_span.end_offset]['value']
            event_mention_id = EventMentionInstanceIdentifierCharOffsetBase(
                event_id.doc_id,
                CharOffsetSpan(sentence_start_char,sentence_end_char),
                CharOffsetSpan(trigger_start_char,trigger_end_char)
            )
            trigger_word = " ".join(sentence_info['token'][event_id.trigger_idx_span.start_offset:event_id.trigger_idx_span.end_offset+1])
            ret.setdefault((doc_path,event_mention_id,event_type,trigger_word),set()).add(("trigger",trigger_start_char,trigger_end_char,True,instance['positive']))
        return ret

    def modify_annotation(self,mongo_instance,session,doc_id_table_name,lemmaizer,event_id,event_type,marking):
        assert isinstance(event_id,EventMentionInstanceIdentifierTokenIdxBase)
        query_key = event_id.reprJSON()
        query_key['eventType'] = event_type
        db_entry = mongo_instance['{}_trigger_marking'.format(session)].find_one(query_key)
        if db_entry is not None:
            db_entry['positive'] = marking
            db_entry['touched'] = True
            mongo_instance['{}_trigger_marking'.format(session)].find_one_and_replace(query_key, db_entry)
        else:
            en_t = mongo_instance[doc_id_table_name].find_one(
                {'docId': event_id.doc_id})
            corpus_name = en_t['corpusName']
            en_t = mongo_instance["{}_sentence_info".format(corpus_name)].find_one(
                {"docId": event_id.doc_id,
                 "sentenceId": event_id.sentence_id
                 })
            sentence_info = en_t['sentenceInfo']
            trigger_word = " ".join(sentence_info['token'][
                                    event_id.trigger_idx_span.start_offset:event_id.trigger_idx_span.end_offset + 1])
            lemma = lemmaizer.get_lemma(trigger_word,True if " "in trigger_word else False)

            en = copy.deepcopy(query_key)
            en['trigger'] = lemma
            en['positive'] = marking
            en['touched'] = True
            mongo_instance['{}_trigger_marking'.format(session)].insert_one(en)

    def annotated_sentence_getter(self,mongo_db,event_type,ontology_event_type_id,session,triggers,DOCID_TABLE,ADJUDICATED_SENTENCE_TABLE_NAME,ANNOTATED_INSTANCE_TABLE_NAME):

        ret = []
        for trigger in triggers:
            lemma = trigger.get('trigger', None)
            postag = trigger.get('postag', None)
            blacklist = trigger.get('blacklist', [])
            blacklist = {EventMentionInstanceIdentifierTokenIdxBase.fromJSON(item) for item in blacklist}
            sentences_ret = []
            from_session_marking_table = mongo_db['{}_trigger_marking'.format(session)].find(
                {'trigger': lemma, "eventType": event_type, "positive": {'$in': [Marking.POSITIVE, Marking.NEGATIVE]}})
            from_adjudicated_table = mongo_db[ANNOTATED_INSTANCE_TABLE_NAME].find(
                {'trigger': lemma, 'eventType': ontology_event_type_id, 'positive': {'$in': [Marking.POSITIVE, Marking.NEGATIVE]}})
            filtered_session_marking_table = list(filter(lambda x: EventMentionInstanceIdentifierTokenIdxBase.fromJSON(x) not in blacklist, from_session_marking_table))
            blacklist.update({EventMentionInstanceIdentifierTokenIdxBase.fromJSON(item) for item in filtered_session_marking_table})
            filtered_adjudicated_table = list(filter(lambda x: EventMentionInstanceIdentifierTokenIdxBase.fromJSON(x) not in blacklist, from_adjudicated_table))
            annotated_records = filtered_session_marking_table + filtered_adjudicated_table
            if annotated_records is None or len(annotated_records) < 1:
                continue
            for s in annotated_records:
                candidate_1 = mongo_db[ADJUDICATED_SENTENCE_TABLE_NAME].find_one(
                    {'docId': s['docId'], 'sentenceId': s['sentenceId']})
                if candidate_1 is None:
                    doc_id_en = mongo_db[DOCID_TABLE].find_one({'docId':s['docId']})
                    if doc_id_en is None:
                        raise ValueError("Cannot find sentence record")
                    candidate_1 = mongo_db['{}_sentence_info'.format(doc_id_en['corpusName'])].find_one(
                        {'docId': s['docId'], 'sentenceId': s['sentenceId']})
                    if candidate_1 is None:
                        raise ValueError("Cannot find sentence record")
                child_sentence = TriggerArgumentInstance.fromMongoDB(s,candidate_1['sentenceInfo']
                )
                sentences_ret.append(child_sentence)
            ret.append({'key': lemma, 'type': 'trigger',
                        'aux': {'trigger': lemma, 'trigger_postag': postag, 'blacklist': list(), 'touched': False},
                        'children': sentences_ret})
        return ret

    def unannotated_sentence_getter(self,mongo_db,session,triggers,limit,corpus_name,event_type=None):
        # @hqiu. blacklist_from_user exist because we can play some frontend trick to filter out duplicate sentence
        ret = []
        SENTENCE_TABLE_NAME = '{}_sentence_info'.format(corpus_name)
        blacklist_for_annotated_sentence_from_mongodb = set()
        if event_type is not None:
            for i in mongo_db['{}_trigger_marking'.format(session)].find(
                    {'eventType': event_type, 'positive': {'$in': [True, False]}}):
                blacklist_for_annotated_sentence_from_mongodb.add(EventMentionInstanceIdentifierTokenIdxBase.fromJSON(i))
        for trigger in triggers:
            lemma = trigger.get('trigger', None)
            postag = trigger.get('postag', None)
            user_blacklist = trigger.get('blacklist', [])
            full_text_search_str = trigger.get('fullTextSearchStr',"")
            sentences_ret = []
            blacklist = set()
            blacklist.update(blacklist_for_annotated_sentence_from_mongodb)
            blacklist.update({EventMentionInstanceIdentifierTokenIdxBase.fromJSON(item) for item in user_blacklist})
            blacklist_simple = set(((item.doc_id, item.sentence_id) for item in blacklist))
            candidates_from_working_corpus_table = list(filter(lambda x: EventMentionInstanceIdentifierTokenIdxBase.fromJSON(x) not in blacklist,mongo_db["{}_trigger_info".format(corpus_name)].find({'trigger': lemma})))
            if full_text_search_str is not None and len(full_text_search_str) > 0:
                candidates_from_full_text_search = list(
                    filter(lambda x: (x['docId'], x['sentenceId']) not in blacklist_simple, mongo_db[SENTENCE_TABLE_NAME].find({
                        '$text': {'$search': full_text_search_str}
                    }, {'docId': 1, 'sentenceId': 1})))
                candidates_from_full_text_search = set((i['docId'], i['sentenceId']) for i in candidates_from_full_text_search)
            else:
                candidates_from_full_text_search = set((i['docId'], i['sentenceId']) for i in candidates_from_working_corpus_table)

            candidates = list()
            for i in candidates_from_working_corpus_table:
                if (i['docId'], i['sentenceId']) in candidates_from_full_text_search:
                    candidates.append(i)

            candidates = candidates[:min(len(candidates), limit)]
            for s in candidates:
                sentence_info = mongo_db[SENTENCE_TABLE_NAME].find_one({'docId': s['docId'], 'sentenceId': s['sentenceId']})[
                    'sentenceInfo']
                child_sentence = TriggerArgumentInstance.fromMongoDB(s,sentence_info)
                sentences_ret.append(child_sentence)
            ret.append({'key': lemma, 'type': 'trigger', 'aux': {'trigger': lemma, 'trigger_postag': postag,
                                                                 'blacklist': list(),'touched': False}, 'children': sentences_ret})
        return ret