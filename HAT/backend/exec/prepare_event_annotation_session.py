import json
import os
import shutil
import sys

current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

from utils.lemmaizer import Lemmaizer
from ontology import internal_ontology


try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

import pymongo

def bootstrap_s3_clusters_from_ontology(yaml_file,json_file,lemmaizer_file_path,output_session_folder):
    shutil.rmtree(output_session_folder,ignore_errors=True)
    os.makedirs(output_session_folder,exist_ok=True)
    output_file = os.path.join(output_session_folder,'s3clusters.json')
    lemmaizer = Lemmaizer(lemmaizer_file_path)
    ui_type_to_ontology_mapping_file = os.path.join(output_session_folder,'ui_event_type_to_ontology_event_type_mapping.json')
    internal_ontology_tree_root1, internal_node_id_to_node_mapping1 = internal_ontology.build_internal_ontology_tree(
        yaml_file, json_file)

    s3clusters = list()
    event_type_mapping = dict()
    for type_id, event_type_nodes in internal_node_id_to_node_mapping1.items():
        for event_type_node in event_type_nodes:
            event_type_path_str = internal_ontology.return_node_name_joint_path_str(event_type_node)
            event_type_mapping[type_id] = event_type_path_str
            if "/Event" not in event_type_path_str:
                continue

            trigger_trimmed_set = set()
            for i in event_type_node.exemplars:
                text = i.text.strip()
                trigger_trimmed_set.add(lemmaizer.get_lemma(text, True if " " in text else False))

            cur_list = list()
            for i in trigger_trimmed_set:
                cur_list.append({'key': i, 'type': 'trigger', 'children': list(),
                                 'aux': {'blacklist': list(), 'trigger': i, 'trigger_postag': None}})
            s3clusters.append({'key': event_type_path_str, 'type': 'cluster', 'children': cur_list})

    s3clusters = sorted(s3clusters,key=lambda x:x['key'],reverse=True)

    with open(output_file,'w') as fp:
        json.dump({'clusters':s3clusters},fp,indent=4,sort_keys=True)
    with open(ui_type_to_ontology_mapping_file,'w') as fp:
        json.dump({val:key for key,val in event_type_mapping.items()},fp,indent=4,sort_keys=True)
    with open(ui_type_to_ontology_mapping_file,'w') as fp:
        json.dump({},fp,indent=4,sort_keys=True)
    return event_type_mapping

def activate_working_corpus(config_ins,session):
    with open(os.path.join(config_ins.FOLDER_SESSION(session),'working_corpora.json'),'w') as fp:
        json.dump(config_ins.CORPORA_FOR_UI_BOOTSTRAPING,fp)

def prepare_session_marking_table(mongo_instance,session):
    # _trigger_marking
    mongo_instance.drop_collection('{}_trigger_marking'.format(session))
    mongo_instance['{}_trigger_marking'.format(session)].create_index(
        [("trigger", pymongo.ASCENDING)], background=True)
    mongo_instance['{}_trigger_marking'.format(session)].create_index(
        [("eventType", pymongo.ASCENDING)], background=True)
    mongo_instance['{}_trigger_marking'.format(session)].create_index(
        [("positive", pymongo.ASCENDING)], background=True)

if __name__ == "__main__":
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join(project_root, "config_default.json"))
    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
    prepare_session_marking_table(mongo_instance,c.init_session_id)
    bootstrap_s3_clusters_from_ontology(c.EVENT_YAML_FILE_PATH,c.DATA_EXAMPLE_JSON_FILE_PATH,c.LEMMA_FILE_PATH,c.FOLDER_SESSION(c.init_session_id))
    activate_working_corpus(c, c.init_session_id)
