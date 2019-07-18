import sys,os
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))


try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

import pymongo

def create_index_for_sentence_db(mongodb_instance,sentence_table_name):
    mongodb_instance[sentence_table_name].create_index(
        [("docId", pymongo.ASCENDING), ("sentenceId", pymongo.ASCENDING)], background=True)

def create_index_for_docid_db(mongodb_instance,docid_table_name):
    mongodb_instance[docid_table_name].create_index([("docId", pymongo.ASCENDING)], background=True)
    mongodb_instance[docid_table_name].create_index([("corpusName", pymongo.ASCENDING)], background=True)

def create_index_for_adjudicated_pool(mongodb_instance,adjudicated_table_name):
    mongodb_instance[adjudicated_table_name].create_index([("trigger", pymongo.ASCENDING),("eventType", pymongo.ASCENDING),("positive", pymongo.ASCENDING)], background=True)
    mongodb_instance[adjudicated_table_name].create_index([("docId", pymongo.ASCENDING),
                                                           ("sentenceId", pymongo.ASCENDING),
                                                           ("triggerSentenceTokenizedPosition", pymongo.ASCENDING),
                                                           ("triggerSentenceTokenizedEndPosition", pymongo.ASCENDING),
                                                           ("eventType", pymongo.ASCENDING)], background=True)

def dropping_existing_table(mongbdb_instance,docid_table_name,sentence_table_name,adjudicated_table_name):
    mongbdb_instance.drop_collection(docid_table_name)
    mongbdb_instance.drop_collection(sentence_table_name)
    mongbdb_instance.drop_collection(adjudicated_table_name)


if __name__ == "__main__":

    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join(project_root, "config_default.json"))
    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
    dropping_existing_table(mongo_instance,c.DOCID_TABLE_NAME,c.ADJUDICATED_SENTENCE_TABLE_NAME,c.ANNOTATED_INSTANCE_TABLE_NAME)
    create_index_for_sentence_db(mongo_instance,c.ADJUDICATED_SENTENCE_TABLE_NAME)
    create_index_for_docid_db(mongo_instance, c.DOCID_TABLE_NAME)
    create_index_for_adjudicated_pool(mongo_instance, c.ANNOTATED_INSTANCE_TABLE_NAME)