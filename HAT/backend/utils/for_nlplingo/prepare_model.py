import os,sys,json,shutil
from utils.from_nlplingo.domain_ontology import read_domain_ontology_file

## @ Legacy code!!! Please use prepare_nlplingo_model_in_pipeline when available

def prepare_nlplingo_event_and_argument_model_folder(trigger_model_path,array_of_event_arg_domain_ontology_path,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    domain_ontology = read_domain_ontology_file(os.path.join(trigger_model_path,'domain_ontology.txt'),'ui')
    event_types = [i for i in domain_ontology.event_types]
    event_types = list(filter(lambda x:x != "None",event_types))
    extractors = list()
    for idx,event_arg_domain_ontology_path in enumerate(array_of_event_arg_domain_ontology_path):
        output_buf = list()
        domain_ontology_arg = read_domain_ontology_file(
            os.path.join(event_arg_domain_ontology_path, 'domain_ontology.txt'), 'ui')
        event_arg_roles = [i for i in domain_ontology_arg.event_roles]
        event_arg_roles = list(filter(lambda x: x != "None", event_arg_roles))
        for event_type in event_types:
            output_buf.append("<Event type=\"{}\">".format(event_type))
            for i in event_arg_roles:
                output_buf.append("<Role>{}</Role>".format(i))
            output_buf.append("</Event>")
        entity_types = [i for i in domain_ontology_arg.entity_types]
        entity_types = list(filter(lambda x: x != "None", entity_types))
        for entity_type in entity_types:
            output_buf.append("<Entity type=\"{}\">".format(entity_type))

        with open(os.path.join(output_folder,'domain_ontology_{}.txt'.format(idx)),'w') as fp:
            for i in output_buf:
                fp.write("{}\n".format(i))
        shutil.copy(os.path.join(event_arg_domain_ontology_path,'argument.hdf'),os.path.join(output_folder,'argument_{}.hdf'.format(idx)))
        extractors.append({
            'model_type':'event-argument_cnn',
            'domain':'general',
            'domain_ontology':os.path.realpath(os.path.join(output_folder,'domain_ontology_{}.txt'.format(idx))),
            'model_file':os.path.realpath(os.path.join(output_folder,'argument_{}.hdf'.format(idx)))
        })
    with open(os.path.join(output_folder,'decoding.json'),'w') as fp:
        json.dump({"extractors":extractors},fp,indent=4,sort_keys=True)
    shutil.copy(os.path.join(trigger_model_path,'domain_ontology.txt'),output_folder)
    shutil.copy(os.path.join(trigger_model_path,'trigger.hdf'),output_folder)

if __name__ == "__main__":
    event_domain_ontology_file_path = "/home/hqiu/massive/wm_m12_wm_intervention_p5/Hume/nlplingo_out"
    # array_of_event_arg_types = [["Actor","Time","Place"],["Time","Place","Active","Affected","Artifact"],["DestinationLocation","SourceLocation"]]
    array_of_event_arg_domain_ontology_path = [
        "/nfs/raid66/u14/users/jfaschin/runjobs/expts/nlplingo_tf/ace_09062018-role_verify_v2",
        "/home/hqiu/Public/wm_m12_hackathon_arg_p2/0/nlplingo_out",
        # "/home/hqiu/massive/wm_event_argument_location_p1/0/nlplingo_out",
    ]
    output_folder = "/home/hqiu/massive/wm_m12_wm_intervention_p5/Hume/nlplingo_out"
    prepare_nlplingo_event_and_argument_model_folder(event_domain_ontology_file_path, array_of_event_arg_domain_ontology_path, output_folder)