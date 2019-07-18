import json
import os
import sys

current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path, os.path.pardir, os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path, os.path.pardir, os.path.pardir,os.path.pardir))


from ontology.internal_ontology import build_internal_ontology_tree_without_exampler, return_node_name_joint_path_str
import logging
from nlp.common import Marking

logger = logging.getLogger(__name__)


def convert_event_types_to_ontology_node_id_with_type2_whitelist(yaml_path, care_list, parsed_dict_from_mongo):
    # @ hqiu in this case. Event type resolving is very straight forward. But we don't want triggers to use their collapsed
    # type. So we return that dictionary independently
    # It's implementing NLPLINGOSerializerJOSHUA2GEN
    parent_node_to_blacklisted_node = dict()
    with open(care_list, 'r') as fp:
        for i in fp:
            i = i.strip()
            care_type, *blacklist_elems = i.split(",")
            care_type = care_type.strip()
            blacklist_elems = set([i.strip() for i in blacklist_elems])
            parent_node_to_blacklisted_node.setdefault(care_type, set()).update(blacklist_elems)

    ontology_tree_root, node_id_to_node_mapping = build_internal_ontology_tree_without_exampler(yaml_path)
    node_id_to_slash_like_path_mapping = dict()
    slash_like_path_to_node_id_mapping = dict()
    for node_id, nodes in node_id_to_node_mapping.items():
        for node in nodes:
            node_id_to_slash_like_path_mapping.setdefault(node_id, set()).add(return_node_name_joint_path_str(node))
            slash_like_path_to_node_id_mapping[return_node_name_joint_path_str(node)] = node_id

    def flatten(root_ptr):
        ret = set()
        ret.add(root_ptr)
        for child in root_ptr.children:
            ret.update(flatten(child))
        return ret

    event_type_id_to_event_type_id_mapping = dict()
    event_type_id_to_bucket_mapping = dict()

    for parent_node_id in parent_node_to_blacklisted_node.keys():
        blacklist_nodes = set(node_id_to_node_mapping[i] for i in parent_node_to_blacklisted_node.get(parent_node_id))
        if parent_node_id not in node_id_to_node_mapping:
            logger.warning("Skipping {} because it's not on the ontology {}".format(parent_node_id,yaml_path))
            continue
        care_nodes = flatten(node_id_to_node_mapping[parent_node_id])
        care_nodes = care_nodes.difference(blacklist_nodes)
        for care_node in care_nodes:
            event_type_id_to_event_type_id_mapping[care_node._id] = node_id_to_node_mapping[parent_node_id]._id
            event_type_id_to_bucket_mapping.setdefault(care_node._id,set()).add(parent_node_id)

    model_to_bucket = dict()
    keyword_dict = dict()
    for k, info_set in parsed_dict_from_mongo.items():
        doc_path, event_mention_id, event_type, trigger_word = k

        if event_type not in slash_like_path_to_node_id_mapping:
            if event_type != "dummy":
                logger.warning(
                    "Due to missing ontology node in {}, we cannot resolve event type {}".format(yaml_path, event_type))
            continue

        original_event_type = slash_like_path_to_node_id_mapping[event_type]

        if original_event_type not in event_type_id_to_event_type_id_mapping:
            logging.info(
                "Omitting entry with event type {} because it's not in the carelist.".format(original_event_type))
            continue
        target_ontology_event_type_id = event_type_id_to_event_type_id_mapping[original_event_type]

        for possible_output_father_node in event_type_id_to_bucket_mapping[target_ontology_event_type_id]:
            model_to_bucket.setdefault(possible_output_father_node, dict()).setdefault(
                (doc_path, event_mention_id, possible_output_father_node, trigger_word), set()).update(info_set)


        for i in info_set:
            _, _, _, _, positive = i
            if positive == Marking.POSITIVE:
                keyword_dict.setdefault(original_event_type, set()).add(trigger_word)
    return model_to_bucket, keyword_dict

def convert_event_types_to_ontology_node_id(yaml_path,parsed_dict_from_mongo):
    ontology_tree_root, node_id_to_node_mapping = build_internal_ontology_tree_without_exampler(yaml_path)
    node_id_to_slash_like_path_mapping = dict()
    slash_like_path_to_node_id_mapping = dict()
    for node_id, nodes in node_id_to_node_mapping.items():
        for node in nodes:
            node_id_to_slash_like_path_mapping[node_id] = return_node_name_joint_path_str(node)
            slash_like_path_to_node_id_mapping[return_node_name_joint_path_str(node)] = node_id
    resolved_parsed_dict_from_mongo = dict()
    for key,val in parsed_dict_from_mongo.items():
        doc_path, event_mention_id, event_type, trigger_word = key
        resolved_event_type = slash_like_path_to_node_id_mapping[event_type]
        resolved_parsed_dict_from_mongo.setdefault((doc_path, event_mention_id, resolved_event_type, trigger_word),set()).update(val)
    return resolved_parsed_dict_from_mongo

def convert_event_types_to_ontology_node_id_with_type1_whitelist(yaml_path, care_list, parsed_dict_from_mongo):
    '''
    @hqiu. For care_list
    different lines means different buckets
    Each element is seperate by ,
    first element is the bucket name
    if element is surrounded by * , it means all its children should be type as its type
    # Means don't handle this line

    A quick example ontology hierachy is like
    /Attack                   contains event instance AT1(Attack),AT2(Attack)
    /Attack/Bombing           contains event instance B1(Bombing),B2(Bombing)
    /Attack/Assaulting        contains event instance AS1(Assaulting),AS2(Assaulting)

    if you write your care_list as:

    batch_1,*Attack*,Bombing
    batch_2,Bombing,*Attack*
    batch_3,*Attack*

    Result is
    batch_1,AT1(Attack),AT2(Attack),B1(Bombing),B2(Bombing),AS1(Attack),AS2(Attack)
    batch_2,AT1(Attack),AT2(Attack),B1(Attack),B2(Attack),AS1(Attack),AS2(Attack)
    batch_3,AT1(Attack),AT2(Attack),B1(Attack),B2(Attack),AS1(Attack),AS2(Attack)

    Summerized
    1) order matters, for left to right in a single batch, the latter(right) types will override former(left)
    2) batches are independent
    :param yaml_path:
    :param care_list: something like /home/hqiu/ld100/CauseEx-pipeline-WM/runtime/carelists/trigger_model.list, explained above
    :param parsed_dict_from_mongo:
    :return:
    '''


    care_list_missing_counter = dict()
    ontology_tree_root, node_id_to_node_mapping = build_internal_ontology_tree_without_exampler(yaml_path)
    node_id_to_slash_like_path_mapping = dict()
    slash_like_path_to_node_id_mapping = dict()
    for node_id, nodes in node_id_to_node_mapping.items():
        for node in nodes:
            node_id_to_slash_like_path_mapping[node_id] = return_node_name_joint_path_str(node)
            slash_like_path_to_node_id_mapping[return_node_name_joint_path_str(node)] = node_id

    care_type_mapping = dict()
    covered_event_type_set = set()

    def flatten(root_ptr):
        ret = set()
        ret.add(root_ptr)
        for child in root_ptr.children:
            ret.update(flatten(child))
        return ret

    with open(care_list, 'r') as fp:
        for i in fp:
            i = i.strip()
            if i.startswith("#"):
                continue
            model_name, *node_ids = i.split(",")
            model_name = model_name.strip()
            for j in node_ids:
                j = j.strip()
                if j.startswith("*") and j.endswith("*"):
                    true_node_name = j[1:-1]
                    if true_node_name not in node_id_to_node_mapping:
                        logger.warning(
                            "Skipping {} because it's not on the ontology {}".format(true_node_name, yaml_path))
                        continue
                    pending_set = flatten(node_id_to_node_mapping[true_node_name])
                    for k in pending_set:
                        en = care_type_mapping.setdefault(model_name,dict())
                        en[k.original_key] = true_node_name
                    covered_event_type_set.update({k.original_key for k in pending_set})
                else:
                    if j not in node_id_to_node_mapping:
                        logger.warning(
                            "Skipping {} because it's not on the ontology {}".format(j, yaml_path))
                        continue
                    en = care_type_mapping.setdefault(model_name, dict())
                    en[j] = j
                    covered_event_type_set.add(j)

    model_to_bucket = dict()
    event_type_id_to_bucket_mapping = dict()
    for model_name in care_type_mapping.keys():
        care_nodes_mapping = care_type_mapping[model_name]
        for care_node in care_nodes_mapping.keys():
            event_type_id_to_bucket_mapping.setdefault(care_node,set()).add(model_name)

    for k, info_set in parsed_dict_from_mongo.items():
        doc_path, event_mention_id, event_type, trigger_word = k

        if event_type not in slash_like_path_to_node_id_mapping:
            if event_type != "dummy":
                logger.warning(
                    "Due to missing ontology node in {}, we cannot resolve event type {}".format(yaml_path, event_type))
            continue

        original_event_type = slash_like_path_to_node_id_mapping[event_type]
        if original_event_type not in covered_event_type_set:
            care_list_missing_counter[original_event_type] = care_list_missing_counter.get(original_event_type,0)+1
            continue

        for pending_bucket in event_type_id_to_bucket_mapping[original_event_type]:
            target_ontology_event_type_id = care_type_mapping[pending_bucket][original_event_type]

            model_to_bucket.setdefault(pending_bucket, dict()).setdefault(
                (doc_path, event_mention_id, target_ontology_event_type_id, trigger_word), set()).update(info_set)
    logger.warning("Due to missing entry on carelist, or internal ontology yaml, we're dropping {}".format(json.dumps(care_list_missing_counter)))
    return model_to_bucket


if __name__ == "__main__":
    pass