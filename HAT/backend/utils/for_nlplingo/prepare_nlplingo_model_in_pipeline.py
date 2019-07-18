import os,sys,json,shutil,copy,logging
current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir,os.path.pardir))
sys.path.append(project_root)
from utils.from_nlplingo.domain_ontology import read_domain_ontology_file

"""
Get dot product of domain_ontology from trigger model, and domain_ontology for argument model.
All outputs are sym links
"""


def generate_decoding_param_template(trigger_train_param, generic_argument_model_folders, argument_model_path_to_argument_model_training_param):
    trigger_train_param = copy.deepcopy(trigger_train_param['extractors'][0])
    trigger_train_param['domain_ontology'] = "+nn_model_dir+/domain_ontology.txt"
    trigger_train_param['model_file'] = "+nn_model_dir+/trigger.hdf"
    defualt_temp = {
        "trigger.restrict_none_examples_using_keywords": False,
        "data":
            {
                "test": {
                    "features": "",
                    "filelist": "+filelist_input+"
                }
            },
        "embeddings": {
            "embedding_file": "+dependencies_root+/nlplingo/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.utf8",
            "missing_token": "the",
            "none_token": ".",
            "vector_size": 400,
            "vocab_size": 251236,
            "none_token_index": 0
        },
        "extractors": [],
        "negative_trigger_words": "+dependencies_root+/nlplingo/negative_words",
        "predictions_file": "+output_dir+/prediction.json"
    }
    defualt_temp["extractors"] = list()
    defualt_temp["extractors"].append(trigger_train_param)
    for idx,generic_argument_model_folder in enumerate(generic_argument_model_folders):
        argument_extractor_dic = argument_model_path_to_argument_model_training_param[generic_argument_model_folder]
        argument_extractor_dic = copy.deepcopy(argument_extractor_dic['extractors'][0])
        argument_extractor_dic['domain_ontology'] = "+nn_model_dir+/domain_ontology_{}.txt".format(idx)
        argument_extractor_dic['model_file'] = "+nn_model_dir+/argument_{}.hdf".format(idx)
        defualt_temp["extractors"].append(argument_extractor_dic)
    return defualt_temp

def main(trigger_model_folders,generic_argument_model_folders,output_path):
    # First pass, get all possible entity type
    possible_entity_types = set()
    # possible_entity_subtypes = set()

    trigger_train_param = None
    for trigger_model_folder in trigger_model_folders:
        domain_ontology = read_domain_ontology_file(os.path.join(trigger_model_folder,'domain_ontology.txt'),'ui')
        entity_types = [i for i in domain_ontology.entity_types]
        entity_types = list(filter(lambda x: x != "None", entity_types))
        possible_entity_types.update(entity_types)
        # entity_subtypes = [i for i in domain_ontology.entity_subtypes]
        # entity_subtypes = list(filter(lambda x: x != "None", entity_subtypes))
        # possible_entity_subtypes.update(entity_subtypes)
        if trigger_train_param is not None:
            logging.warning("We're not supporting multiple trigger parameter. Trigger param will be override now")
        with open(os.path.join(trigger_model_folder,'train.params.json')) as fp:
            trigger_train_param = json.load(fp)
    argument_model_path_to_potential_argument_roles = dict()
    argument_model_path_to_argument_hdf_path = dict()
    argument_model_path_to_argument_model_training_param = dict()

    for generic_argument_model_folder in generic_argument_model_folders:
        domain_ontology = read_domain_ontology_file(os.path.join(generic_argument_model_folder,'domain_ontology.txt'),'ui')
        entity_types = [i for i in domain_ontology.entity_types]
        entity_types = list(filter(lambda x: x != "None", entity_types))
        possible_entity_types.update(entity_types)
        # entity_subtypes = [i for i in domain_ontology.entity_subtypes]
        # entity_subtypes = list(filter(lambda x: x != "None", entity_subtypes))
        # possible_entity_subtypes.update(entity_subtypes)
        event_arg_roles = [i for i in domain_ontology.event_roles]
        event_arg_roles = list(filter(lambda x: x != "None", event_arg_roles))
        argument_model_path_to_potential_argument_roles.setdefault(generic_argument_model_folder,set()).update(event_arg_roles)
        argument_model_path_to_argument_hdf_path[generic_argument_model_folder] = os.path.join(generic_argument_model_folder,'argument.hdf')
        with open(os.path.join(generic_argument_model_folder,'train.param.json')) as fp:
            argument_model_path_to_argument_model_training_param[generic_argument_model_folder] = json.load(fp)

    # Second pass, Assemble!
    shutil.rmtree(output_path,ignore_errors=True)
    os.makedirs(output_path,exist_ok=True)
    output_model_folders = list()

    for trigger_model_folder in trigger_model_folders:
        base_name = os.path.basename(trigger_model_folder)
        current_output_path = os.path.join(output_path,base_name)
        output_model_folders.append(current_output_path)
        os.makedirs(os.path.join(output_path,base_name))
        domain_ontology = read_domain_ontology_file(os.path.join(trigger_model_folder,'domain_ontology.txt'),'ui')
        event_types = [i for i in domain_ontology.event_types]
        event_types = list(filter(lambda x:x != "None",event_types))

        # Step 1: trigger model
        with open(os.path.join(current_output_path,'domain_ontology.txt'),'w') as fp:
            for event_type in event_types:
                fp.write("<Event type=\"{}\">\n".format(event_type))
                fp.write("</Event>\n")
            for entity_type in possible_entity_types:
                fp.write("<Entity type=\"{}\">\n".format(entity_type))
            # for entity_subtype in possible_entity_subtypes:
            #     fp.write("<Entity subtype=\"{}\">\n".format(entity_subtype))
        os.symlink(os.path.join(trigger_model_folder,'trigger.hdf'),os.path.join(current_output_path,'trigger.hdf'))

        # Step 2: argument models
        for idx,generic_argument_model_folder in enumerate(generic_argument_model_folders):
            with open(os.path.join(current_output_path,'domain_ontology_{}.txt'.format(idx)),'w') as fp:
                for event_type in event_types:
                    fp.write("<Event type=\"{}\">\n".format(event_type))
                    for arg_type in argument_model_path_to_potential_argument_roles[generic_argument_model_folder]:
                        fp.write("<Role>{}</Role>\n".format(arg_type))
                    fp.write("</Event>\n")
                for entity_type in possible_entity_types:
                    fp.write("<Entity type=\"{}\">\n".format(entity_type))
                # for entity_subtype in possible_entity_subtypes:
                #     fp.write("<Entity subtype=\"{}\">\n".format(entity_subtype))
            os.symlink(os.path.join(generic_argument_model_folder,'argument.hdf'),os.path.join(current_output_path,'argument_{}.hdf'.format(idx)))

    with open(os.path.join(output_path,'model.list'),'w') as fp:
        for idx,i in enumerate(output_model_folders):
            fp.write("{} {}\n".format(os.path.basename(os.path.realpath(os.path.join(i))),i))

    decoding_template = generate_decoding_param_template(trigger_train_param, generic_argument_model_folders, argument_model_path_to_argument_model_training_param)
    with open(os.path.join(output_path,'nlplingo_decoding.par.json'),'w') as fp:
        json.dump(decoding_template,fp,indent=4,sort_keys=True)

if __name__ == "__main__":
    trigger_model_folders = [
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_1",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_2",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_3",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_4",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_5",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/060619/batch_6"
    ]
    generic_argument_model_folders = [
        "/nfs/raid87/u10/shared/Hume/common/ace_arg_model",
        "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models/argbefore053019/batch_1"
    ]
    output_path = "/nfs/raid87/u10/shared/Hume/wm/nlplingo/wm_models_virtual/060619"
    main(trigger_model_folders,generic_argument_model_folders,output_path)

