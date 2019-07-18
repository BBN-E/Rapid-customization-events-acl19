import yaml,json,os,sys,collections

from ontology.model import InternalOntologyTreeNode,Trigger_Item_Model

def dfs_visit_build_internal_ontology_treenode(root,node_name_to_nodes_mapping,current_node_key):
    children = list()
    key_dict = dict()
    for dct in root:
        for key, value in dct.items():
            if key.startswith("_"):
                key_dict[key] = value
            else:
                children.append(dfs_visit_build_internal_ontology_treenode(value,node_name_to_nodes_mapping,key))
    new_internal_ontology_node = InternalOntologyTreeNode(**key_dict)
    new_internal_ontology_node.parent = None
    new_internal_ontology_node.children = children
    new_internal_ontology_node.exemplars = set()
    for i in children:
        i.parent = new_internal_ontology_node
    new_internal_ontology_node.original_key = current_node_key
    node_name_to_nodes_mapping.setdefault(current_node_key,set()).add(new_internal_ontology_node)
    return new_internal_ontology_node

def build_internal_ontology_tree(yaml_path,exampler_path):
    with open(yaml_path,'r') as fp:
        y = yaml.load(fp)
    node_name_to_nodes_mapping = dict()
    yaml_root = y[0]['Event']
    ontology_tree_root = dfs_visit_build_internal_ontology_treenode(yaml_root,node_name_to_nodes_mapping,"Event")

    with open(exampler_path,'r') as fp:
        exemplars = json.load(fp)
    for event_type_id,data_obj in exemplars.items():
        if event_type_id in node_name_to_nodes_mapping:
            node_name_to_nodes_mapping[event_type_id].exemplars.update({Trigger_Item_Model(j['trigger']['POS'],j['trigger']['text'],j['trigger']['wn_sense']) for j in data_obj['exemplars']})
            node_name_to_nodes_mapping[event_type_id].exemplars_aux = dict()
            for k,v in data_obj.items():
                if k == "exemplars":
                    continue
                node_name_to_nodes_mapping[event_type_id].exemplars_aux[k] = v
    return ontology_tree_root,node_name_to_nodes_mapping

def build_internal_ontology_tree_without_exampler(yaml_path):
    with open(yaml_path,'r') as fp:
        y = yaml.load(fp)
    node_name_to_nodes_mapping = dict()
    yaml_root = y[0]['Event']
    ontology_tree_root = dfs_visit_build_internal_ontology_treenode(yaml_root,node_name_to_nodes_mapping,"Event")
    return ontology_tree_root,node_name_to_nodes_mapping

def get_first_available_key(node_id_prefix,exist_id_set):
    for i in range(1,1000):
        if "{}_{}".format(node_id_prefix, str(i).zfill(3)) not in exist_id_set:
            return "{}_{}".format(node_id_prefix, str(i).zfill(3))
    return "{}_{}".format(node_id_prefix, 999)

def build_internal_ontology_tree_from_hume_ontology_tree(hume_ontology_tree_root,exist_id_set,node_id_to_node_mapping,previous_hume_root_id_to_new_internal_root_id_mapping):

    current_type = hume_ontology_tree_root.original_key.split("/")[-1]
    current_type = current_type.replace(" ","_")
    next_available_id = get_first_available_key(current_type,exist_id_set)
    exist_id_set.add(next_available_id)

    key_dict = {
        '_id':next_available_id,
        '_source':['NN','HUME: hume/{}'.format("/".join(hume_ontology_tree_root.original_key.split("/")[1:]))],
        '_description':'NA',
        '_examples':['NA']
    }
    new_internal_ontology_node = InternalOntologyTreeNode(**key_dict)
    new_internal_ontology_node.original_key = next_available_id
    children = list()
    for i in hume_ontology_tree_root.children:
        children.append(build_internal_ontology_tree_from_hume_ontology_tree(i,exist_id_set,node_id_to_node_mapping,previous_hume_root_id_to_new_internal_root_id_mapping))
    new_internal_ontology_node.children = children
    new_internal_ontology_node.exemplars = {Trigger_Item_Model(None,i,None) for i in hume_ontology_tree_root.exemplars}
    node_id_to_node_mapping[next_available_id] = new_internal_ontology_node
    previous_hume_root_id_to_new_internal_root_id_mapping[hume_ontology_tree_root.original_key] = new_internal_ontology_node
    return new_internal_ontology_node

def serialize_yaml(internal_ontology_root):
    ret_arr = list()
    for k,v in internal_ontology_root.__dict__.items():
        if k.startswith("_"):
            ret_arr.append({k:v})
    for child in internal_ontology_root.children:
        ret_dict,child_root_name = serialize_yaml(child)
        ret_arr.extend(ret_dict)
    return [{internal_ontology_root.original_key:ret_arr}],internal_ontology_root.original_key

def serialize_exemplars(internal_ontology_root):
    ret_dict = dict()
    if internal_ontology_root.__dict__.get('exemplars',None) is not None:
        d = dict()
        d.update(internal_ontology_root.__dict__.get("exemplars_aux",dict()))
        d["exemplars"] = ([{"trigger":{"POS":i.POS,"text":i.text,"wn_sense":i.wn_sense}} for i in internal_ontology_root.exemplars])
        ret_dict[internal_ontology_root._id] = d
    for child in internal_ontology_root.children:
        ret_dict.update(serialize_exemplars(child))
    return ret_dict

def return_node_name_joint_path_str(internal_ontology_node):
    buf = list()
    buf.append(internal_ontology_node.original_key)
    ptr = internal_ontology_node.parent
    while ptr:
        buf.append(ptr.original_key)
        ptr = ptr.parent
    return "/"+"/".join(buf[::-1])

def return_node_id_joint_path_str(internal_ontology_node):
    buf = list()
    buf.append(internal_ontology_node._id)
    ptr = internal_ontology_node.parent
    while ptr:
        buf.append(ptr._id)
        ptr = ptr.parent
    return "/"+"/".join(buf[::-1])


def find_lowest_nn_node(internal_ontology_node):
    if internal_ontology_node is None:
        return None
    for property in internal_ontology_node._source:
        if property == "NN":
            return internal_ontology_node
    return find_lowest_nn_node(internal_ontology_node.parent)