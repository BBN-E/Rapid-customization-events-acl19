import yaml



def dfs_tree_visit(node,current_stack,result_dict):
    if isinstance(node,list):
        for child in node:
            dfs_tree_visit(child,current_stack,result_dict)
    elif isinstance(node,str):
        current_stack.append(node)
        result_dict[node] = "/" + "/".join(current_stack)
        current_stack.pop()
    elif isinstance(node,dict):
        for k,v in node.items():
            current_stack.append(k)
            dfs_tree_visit(v,current_stack,result_dict)
            current_stack.pop()
    else:
        print(node)

def main(yaml_path):
    with open(yaml_path) as fp:
        ontology = yaml.load(fp)
    current_stack = list()
    result_dict= dict()
    dfs_tree_visit(ontology,current_stack,result_dict)
    for k,v in result_dict.items():
        print("{}: {}".format(k,v))



if __name__ == "__main__":
    hume_yaml = "/home/hqiu/ld100/Ontologies/performer_ontologies/hume_ontology.yaml"
    wm_yaml = "/home/hqiu/ld100/Ontologies/wm.yml"
    # main(hume_yaml)
    main(wm_yaml)