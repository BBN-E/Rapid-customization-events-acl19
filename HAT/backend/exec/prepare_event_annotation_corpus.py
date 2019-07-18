import shutil,os,json,subprocess,sys

current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))

import pymongo



try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

import subprocess
import os


def cbc_trigger_sentence_info_generator(BIN_CBC_TRIGGER_SENTENCEINFO_GENERATOR,serifxml_list_path,stopword_list_path,cbc_s3_trigger_info_path,cbc_s3_sentence_info_path):
    david_prog = "{} {} {} {} {}".format(BIN_CBC_TRIGGER_SENTENCEINFO_GENERATOR,
                                         serifxml_list_path, stopword_list_path, cbc_s3_trigger_info_path,cbc_s3_sentence_info_path)
    subprocess.check_call([david_prog],shell=True)




def trigger_frequency_filter(cbc_s3_trigger_info_path,threshold):
    count_map = dict()
    print("Starting filtering trigger by threshold {}".format(threshold))
    with open(cbc_s3_trigger_info_path) as fp:
        for row in fp:
            j = json.loads(row)
            j['docId'] = str(j['docId'])
            j['sentenceId'] = str((j['sentenceId']['key'],j['sentenceId']['value']))
            count_map[(j['trigger'],j['triggerPosTag'])] = count_map.get((j['trigger'],j['triggerPosTag']),0)+ 1
    ret_set = set()
    for word_tuple,cnt in count_map.items():
        if cnt >= threshold:
            ret_set.add(word_tuple)
    print("Finish filtering trigger")
    return ret_set

def prepare_corpus_trigger_candidate_table_from_cbc_s3_trigger_into(cbc_s3_trigger_info_path,corpus_name,mongodb_instance,trigger_whitelist_set=None):
    mongodb_instance.drop_collection('{}_trigger_info'.format(corpus_name))
    adding_arr = list()
    print("Checking trigger file")
    cnt = 0
    with open(cbc_s3_trigger_info_path) as fp:
        for row in fp:
            j = json.loads(row)
            j['sentenceId'] = str((j['sentenceId']['key'], j['sentenceId']['value']))
            if isinstance(trigger_whitelist_set,set) and len(trigger_whitelist_set) > 0:
                if (j['trigger'],j['triggerPosTag']) not in trigger_whitelist_set:
                    continue
            adding_arr.append(j)
            cnt += 1
            if cnt % 10000 == 0:
                print("Examined {} lines of triggers.".format(cnt))
    print("Start insert trigger candidate into mongo")
    mongodb_instance['{}_trigger_info'.format(corpus_name)].insert_many(adding_arr)
    print("Creating Index for triggers")
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("docId", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("sentenceId", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("trigger", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("triggerSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("triggerSentenceTokenizedEndPosition", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("timeSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)
    mongodb_instance['{}_trigger_info'.format(corpus_name)].create_index(
        [("locationSentenceTokenizedPosition", pymongo.ASCENDING)], background=True)


def prepare_corpus_sentence_table_from_cbc_s3_sentence_into(cbc_s3_sentence_info_path,corpus_name,mongo_instance):
    docId_set = set()
    mongo_instance.drop_collection('{}_sentence_info'.format(corpus_name))
    print("Creating index for sentence id search")
    mongo_instance['{}_sentence_info'.format(corpus_name)].create_index(
        [("docId", pymongo.ASCENDING), ("sentenceId", pymongo.ASCENDING)], background=True)
    print("Creating index for full text search")
    mongo_instance['{}_sentence_info'.format(corpus_name)].create_index(
        [("fullSentenceText", pymongo.TEXT)], background=True)
    with open(cbc_s3_sentence_info_path, 'r') as fp:
        for i in fp:
            i = i.strip()
            j = json.loads(i)
            docId_set.add(j['docId'])
    sentence_info_pending_list = list()
    print("Checking sentence file")
    cnt = 0
    with open(cbc_s3_sentence_info_path) as fp:
        for row in fp:
            j = json.loads(row)
            j['sentenceId'] = str((j['sentenceId']['key'], j['sentenceId']['value']))
            sentence_info_pending_list.append(j)
            cnt += 1
            if cnt % 10000 == 0:
                print("Examined {} lines of sentences.".format(cnt))
    existing_sentence_set = set()
    for idx,i in enumerate(sentence_info_pending_list):
        if (i['docId'],i['sentenceId']) in existing_sentence_set:
            continue
        else:
            en = mongo_instance['{}_sentence_info'.format(corpus_name)].find_one({'docId':i['docId'],'sentenceId':i['sentenceId']})
            if en is None:
                mongo_instance['{}_sentence_info'.format(corpus_name)].insert_one(i)
                # print("Line {} Inserted".format(idx))
            else:
                pass
                # print("Line {} Existed".format(idx))
            existing_sentence_set.add((i['docId'],i['sentenceId']))
            if idx % 10000 == 0:
                print("Inserted {} lines of sentences.".format(idx))


def maintain_docid_table(sentence_info_ljson_path,mongo_instance,DOCID_TABLE_NAME,corpus_name):
    pending_set = set()
    with open(sentence_info_ljson_path,'r') as fp:
        for i in fp:
            i = i.strip()
            j = json.loads(i)
            doc_id = j['docId']
            doc_path = j['docPath']
            pending_set.add((doc_id,doc_path))
    for doc_id,doc_path in pending_set:
        en = mongo_instance[DOCID_TABLE_NAME].find_one({'docId':doc_id})
        if en is None:
            mongo_instance[DOCID_TABLE_NAME].insert_one({'docId':doc_id,'docPath':doc_path,'corpusName':corpus_name})
        else:
            raise ValueError("Currently we intend to insert {} under corpus {}, but it's exist in {}".format(doc_id,corpus_name,en['corpusName']))
                # mongo_instance[DOCID_TABLE_NAME].find_one_and_replace({'docId':doc_id},{'docId':doc_id,'docPath':i,'corpusName':corpus_name})



if __name__ == "__main__":

    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join(project_root,"config_default.json"))


    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)

    for corpus in c.CORPORA_FOR_DB_BOOTSTRAPING:

        desired_cbc_s3_trigger_info_path = corpus.cbc_m3_trigger_info if hasattr(corpus,"cbc_m3_trigger_info") else os.path.realpath(os.path.join(corpus.serifxml_list_path,os.path.pardir,corpus.corpus_name+"_trigger.ljson"))
        desired_cbc_s3_sentence_info_path = corpus.cbc_m3_sentence_info if hasattr(corpus,"cbc_m3_sentence_info") else os.path.realpath(os.path.join(corpus.serifxml_list_path,os.path.pardir,corpus.corpus_name+"_sentence.ljson"))
        print(desired_cbc_s3_trigger_info_path)
        print(desired_cbc_s3_sentence_info_path)
        if (os.path.isfile(desired_cbc_s3_trigger_info_path) is False) or (os.path.isfile(desired_cbc_s3_sentence_info_path) is False):

            cbc_trigger_sentence_info_generator(c.BIN_CBC_TRIGGER_SENTENCEINFO_GENERATOR, corpus.serifxml_list_path,
                                                c.STOPWORD_LIST_PATH,
                                                desired_cbc_s3_trigger_info_path, desired_cbc_s3_sentence_info_path)

        reserved_set = trigger_frequency_filter(desired_cbc_s3_trigger_info_path, c.TRIGGER_CANDIDATE_FREQ_FILTER)
        maintain_docid_table(desired_cbc_s3_sentence_info_path,mongo_instance,c.DOCID_TABLE_NAME,corpus.corpus_name)
        prepare_corpus_sentence_table_from_cbc_s3_sentence_into(desired_cbc_s3_sentence_info_path, corpus.corpus_name, mongo_instance)
        prepare_corpus_trigger_candidate_table_from_cbc_s3_trigger_into(desired_cbc_s3_trigger_info_path, corpus.corpus_name,
                                                                        mongo_instance, reserved_set)

