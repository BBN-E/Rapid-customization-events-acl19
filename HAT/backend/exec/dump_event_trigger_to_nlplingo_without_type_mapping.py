import os,sys,json,re,copy,shutil
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
from event_trigger_data_management.serializer.nlplingo import NLPLINGOSerializerHAOLING
from event_trigger_data_management.resolver.bad_event_type_name_resolver import resolve_slash
from event_trigger_data_management.reader.mongodb import MongoDBReader

def direct_passthrough(mongo_instance,session,DOCID_TABLE_NAME,output_folder):
    mongodb_reader = MongoDBReader(mongo_instance,DOCID_TABLE_NAME,None,None)
    annotated_pool = mongodb_reader.parse_db_session_table_directly_out(session)
    annotated_pool = resolve_slash(annotated_pool)
    docId_to_docPath = mongodb_reader.docId_to_docPath_mapping()
    serialize = NLPLINGOSerializerHAOLING(annotated_pool,docId_to_docPath,output_folder)
    serialize.serialize()

if __name__ == "__main__":
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))

    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
    direct_passthrough(
        mongo_instance,
        c.init_session_id,
        c.DOCID_TABLE_NAME,
        c.NLPLINGO_SPAN_OUTPUT_PATH
    )
