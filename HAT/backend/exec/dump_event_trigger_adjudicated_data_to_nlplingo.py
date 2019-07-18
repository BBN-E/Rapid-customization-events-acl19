import os, sys
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

import pymongo
import argparse
from event_trigger_data_management.serializer.nlplingo import NLPLINGOSerializerHAOLING,NLPLINGOSerializerJOSHUA,NLPLINGOSerializerJOSHUA2GEN
from event_trigger_data_management.reader.mongodb import MongoDBReader
try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

def _serialize_driver(
    mongo_instance,
    DOCID_TABLE_NAME,
    SENTENCE_TABLE_NAME,
    ANNOTATED_INSTANCE_TABLE_NAME,
    ontology_yaml_file,
    output_dir
):
    mongodb_reader = MongoDBReader(
        mongo_instance,
        DOCID_TABLE_NAME,
        SENTENCE_TABLE_NAME,
        ANNOTATED_INSTANCE_TABLE_NAME
    )
    annotated_pool, docId_to_docPath = mongodb_reader.parse_whole_yaml_tree_out_with_auto_resolved_merge(
        ontology_yaml_file
    )
    nlplingo_serializer = NLPLINGOSerializerHAOLING(
        annotated_pool,
        docId_to_docPath,
        output_dir
    )
    nlplingo_serializer.serialize()

if __name__ == "__main__":
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))

    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)

    _serialize_driver(
        mongo_instance,
        c.DOCID_TABLE_NAME,
        c.SENTENCE_TABLE_NAME,
        c.ANNOTATED_INSTANCE_TABLE_NAME,
        c.EVENT_YAML_FILE_PATH,
        c.NLPLINGO_SPAN_OUTPUT_PATH
    )