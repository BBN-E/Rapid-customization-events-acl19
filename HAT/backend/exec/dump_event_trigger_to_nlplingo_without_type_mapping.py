import os
import re
import sys

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
from data_access_layer.trigger_annotation.mongodb import TriggerAnnotationMongoDBAccessLayer
from data_access_layer.common.nlplingo import nlplingo_serializer_flatten_flavor


def direct_passthrough(mongo_instance,session,DOCID_TABLE_NAME,output_folder):
    dao = TriggerAnnotationMongoDBAccessLayer()
    annotations = dao.dump_annotation_out(mongo_instance, session, DOCID_TABLE_NAME)
    nlplingo_serializer_flatten_flavor(annotations, output_folder)

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