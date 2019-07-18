import os,sys,json,re,copy
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

import pymongo
try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus
tuple_stype_re_pattern = re.compile(r"^\((\d+), (\d+)\)$")


def clean_scratch_table(mongo_instance,SCRATCH_TABLE_NAME,SCRATCH_TABLE_NAME_2):
    mongo_instance.drop_collection(SCRATCH_TABLE_NAME)
    mongo_instance[SCRATCH_TABLE_NAME].create_index(
        [("docId", pymongo.ASCENDING)], background=True)
    mongo_instance[SCRATCH_TABLE_NAME].create_index(
        [("sentenceId", pymongo.ASCENDING)], background=True)
    mongo_instance[SCRATCH_TABLE_NAME].create_index(
        [("triggerSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)
    mongo_instance[SCRATCH_TABLE_NAME].create_index(
        [("triggerSentenceTokenizedEndPosition", pymongo.ASCENDING)], background=True)
    mongo_instance[SCRATCH_TABLE_NAME].create_index(
        [("eventType", pymongo.ASCENDING)], background=True)
    mongo_instance.drop_collection(SCRATCH_TABLE_NAME_2)
    mongo_instance[SCRATCH_TABLE_NAME_2].create_index([("docId", pymongo.ASCENDING),("sentenceId",pymongo.ASCENDING)], background=True)

def collution_check(
        mongo_instance,
        session,
        DOCID_TABLE_NAME,
        ANNOTATED_INSTANCE_TABLE_NAME,
        ADJUDICATED_SENTENCE_TABLE_NAME,
        SCRATCH_TABLE_NAME,
        SCRATCH_TABLE_NAME_2,
        ui_type_to_ontology_type_mapping_json_path
):

    docId_to_docInfo = dict()
    with open(ui_type_to_ontology_type_mapping_json_path,'r') as fp:
        ui_type_to_ontology_type_mapping = json.load(fp)

    for i in mongo_instance['{}_trigger_marking'.format(session)]:
        annotation_type = i['eventType']
        try:
            resolved_event_type = ui_type_to_ontology_type_mapping[annotation_type]
        except KeyError as e:
            if annotation_type == "dummy":
                continue
            else:
                raise e
        # resolved_event_type = annotation_type
        if docId_to_docInfo.get(i['docId'], None) is None:
            en = mongo_instance[DOCID_TABLE_NAME].find_one({'docId': i['docId']})
            if en is None:
                raise RuntimeError("Fail to look up entry for docId: {}".format(i['docId']))
            docId_to_docInfo[i['docId']] = en

        sentence_info = mongo_instance[ADJUDICATED_SENTENCE_TABLE_NAME].find_one(
            {
                'docId': i['docId'],
                'sentenceId': i['sentenceId']
            }
        )

        if sentence_info is None:
            sentence_info = mongo_instance['{}_sentence_info'.format(docId_to_docInfo[i['docId']]['corpusName'])].find_one({
                'docId': i['docId'],
                'sentenceId': i['sentenceId']
            })
        if sentence_info is None:
            raise RuntimeError("Fail to look up sentence info for docId:{}, sentenceId:{}".format(i['docId'],i['sentenceId']))

        sentence_key = {'docId':i['docId'],'sentenceId':i['sentenceId']}
        en = mongo_instance[SCRATCH_TABLE_NAME_2].find_one(sentence_key)
        if en is None:
            en2 = mongo_instance['{}_sentence_info'.format(docId_to_docInfo[i['docId']]['corpusName'])].find_one({
                'docId': i['docId'],
                'sentenceId': i['sentenceId']
            })
            try:
                del en2['fullSentenceText']
            except KeyError:
                pass
            del en2['_id']
            mongo_instance[SCRATCH_TABLE_NAME_2].insert_one(en2)


        key = {
            'docId': i['docId'],
            'sentenceId': i['sentenceId'],
            'triggerSentenceTokenizedPosition': i['triggerSentenceTokenizedPosition'],
            'triggerSentenceTokenizedEndPosition': i['triggerSentenceTokenizedEndPosition'],
            'eventType': resolved_event_type
        }
        potential1 = mongo_instance[ANNOTATED_INSTANCE_TABLE_NAME].find_one(key)
        potential2 = mongo_instance[SCRATCH_TABLE_NAME].find_one(key)

        annotation_table_entry = {
            'docId': i['docId'],
            'sentenceId': i['sentenceId'],
            'triggerSentenceTokenizedPosition': i['triggerSentenceTokenizedPosition'],
            'triggerSentenceTokenizedEndPosition': i['triggerSentenceTokenizedEndPosition'],
            'trigger': " ".join(
                sentence_info['sentenceInfo']['token'][i['triggerSentenceTokenizedPosition']:i['triggerSentenceTokenizedEndPosition']+1]
            ),
            "triggerPosTag": i['triggerPosTag'],
            "positive": i['ANNOTATION_{}'.format(annotation_type)],
            "corpusName": docId_to_docInfo[i['docId']]['corpusName'],
            "annotationTaskName": "{}_{}".format(
                docId_to_docInfo[i['docId']]['corpusName'],
                session
            ),
            "eventType": resolved_event_type
        }
        if potential2 is not None:
            print(
                "WARNING: duplicate detected {} because it's in the {}".format(
                    key,
                    SCRATCH_TABLE_NAME
                )
            )
            mongo_instance[SCRATCH_TABLE_NAME].find_one_and_replace(
                key,
                annotation_table_entry
            )
            continue
        if potential1 is not None:
            print(
                "WARNING: duplicate detected {} because it's in the {}".format(
                    key,
                    ANNOTATED_INSTANCE_TABLE_NAME
                )
            )
            pass
        mongo_instance[SCRATCH_TABLE_NAME].insert_one(annotation_table_entry)


def finalize(
        src_mongo_instance,
        dst_mongo_instance,
        ANNOTATED_INSTANCE_TABLE_NAME,
        ADJUDICATED_SENTENCE_TABLE_NAME,
        SCRATCH_TABLE_NAME,
        SCRATCH_TABLE_NAME_2
):
    for i in src_mongo_instance[SCRATCH_TABLE_NAME].find({}):
        annotation_table_entry = copy.deepcopy(i)
        del annotation_table_entry['_id']
        key = {
            'docId': i['docId'],
            'sentenceId': i['sentenceId'],
            'triggerSentenceTokenizedPosition': i['triggerSentenceTokenizedPosition'],
            'triggerSentenceTokenizedEndPosition': i['triggerSentenceTokenizedEndPosition'],
            'eventType': i['eventType']
        }
        potential_entry = dst_mongo_instance[ANNOTATED_INSTANCE_TABLE_NAME].find_one(key)
        if potential_entry is None:
            dst_mongo_instance[ANNOTATED_INSTANCE_TABLE_NAME].insert_one(annotation_table_entry)
        else:
            dst_mongo_instance[ANNOTATED_INSTANCE_TABLE_NAME].find_one_and_replace(key, annotation_table_entry)
    for i in src_mongo_instance[SCRATCH_TABLE_NAME_2].find({}):
        en = dst_mongo_instance[ADJUDICATED_SENTENCE_TABLE_NAME].find_one({'docId':i['docId'],'sentenceId':i['sentenceId']})
        if en is None:
            del i['_id']
            dst_mongo_instance[ADJUDICATED_SENTENCE_TABLE_NAME].insert_one(i)

if __name__ == "__main__":

    from config import ReadAndConfiguration

    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))
    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)



    # ui_type_to_ontology_type_mapping = "/nfs/ld100/u10/hqiu/hume/hmi/backend/customized-event/tmp/wm_m12_wm_intervention_1/ui_event_type_to_ontology_event_type_mapping.json"
    clean_scratch_table(
        mongo_instance,
        c.DB_SCRATCH_TABLE_NAME,
        c.DB_SCRATCH_TABLE_NAME_2
    )

    collution_check(
        mongo_instance,
        c.init_session_id,
        c.DOCID_TABLE_NAME,
        c.ANNOTATED_INSTANCE_TABLE_NAME,
        c.ADJUDICATED_SENTENCE_TABLE_NAME,
        c.DB_SCRATCH_TABLE_NAME,
        c.DB_SCRATCH_TABLE_NAME_2,
        os.path.join(c.FOLDER_SESSION(c.init_session_id),"ui_event_type_to_ontology_event_type_mapping.json")
    )

    finalize(mongo_instance,mongo_instance,c.ANNOTATED_INSTANCE_TABLE_NAME,c.ADJUDICATED_SENTENCE_TABLE_NAME,c.DB_SCRATCH_TABLE_NAME,c.DB_SCRATCH_TABLE_NAME_2)
    # finalize(production_mongo_instance,internal_mongo_instance,ANNOTATED_INSTANCE_TABLE_NAME,SCRATCH_TABLE_NAME)