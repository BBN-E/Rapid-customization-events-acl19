import os, sys, json,shutil,stat,multiprocessing,shlex,subprocess

# MY_SERIFXML_PY_PATH = "/home/hqiu/SVN_PROJECT_ROOT_LOCAL"
# NLPLINGO_ROOT = "/nfs/raid87/u13/nlplingo_bert"
# ANACONDA_ROOT = "/nfs/raid87/u11/users/hqiu/external_dependencies_unmanaged/anaconda"
# CONDA_ENV = "tensorflow-1.5"

current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir,os.path.pardir))
sys.path.append(project_root)

from utils.from_nlplingo import filter_serif_sentence

def multiprocess_wrapper(command):
    try:
        subprocess.check_call(shlex.split(command))
    except:
        import traceback
        traceback.print_exc()

def shrink_serifxml(my_output_folder):
    worker_list = list()
    manager = multiprocessing.Manager()
    with manager.Pool() as pool:
        for folder in os.listdir(my_output_folder):
            if "_shrinked_serifxml" in folder:
                continue
            booking_file = os.path.join(my_output_folder,folder,'argument.span_serif_list')
            enabled_span_file = os.path.join(my_output_folder,folder,'argument.sent_spans')
            output_serif_path = os.path.join(my_output_folder,folder+"_shrinked_serifxml")
            shutil.rmtree(output_serif_path,ignore_errors=True)
            os.makedirs(output_serif_path,exist_ok=True)
            output_booking_file_path = os.path.join(my_output_folder,folder,'argument.span_serif_list.shrinked')
            command_line = "python {} {} {} {} {}".format("{}/utils/from_nlplingo/filter_serif_sentence.py",project_root,booking_file,enabled_span_file,output_serif_path,output_booking_file_path)
            # subprocess.check_call(shlex.split("python {} {} {} {} {}".format("/nfs/raid87/u12/ychan/wm/m12/experiments/100_event_types/batch_1/filter_serif_sentence.py",booking_file,enabled_span_file,output_serif_path,output_booking_file_path)))

            # worker_list.append(pool.apply_async(multiprocess_wrapper,args=(command_line,)))
            worker_list.append(pool.apply_async(filter_serif_sentence,args=(booking_file,enabled_span_file,output_serif_path,output_booking_file_path)))
        for idx,i in enumerate(worker_list):
            print("Waiting for {}".format(len(worker_list)-idx))
            i.wait()
            print(i.get())


def make_json_param(input_folder,embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,output_folder):
    shutil.rmtree(output_folder,ignore_errors=True)
    os.makedirs(output_folder,exist_ok=True)
    shutil.copy(os.path.join(input_folder,'domain_ontology.txt'),os.path.join(output_folder,'domain_ontology.txt'))

    json_param = {
        "trigger.restrict_none_examples_using_keywords": False,
        "data": {
            "train": {"filelist": os.path.join(input_folder, 'argument.span_serif_list.shrinked')},
            "dev": {"filelist": os.path.join(input_folder, 'argument.span_serif_list.shrinked')}
        },
        "embeddings": {
            "embedding_file": embedding_path,
            "missing_token": "the",
            "none_token": ".",
            "vector_size": 400,
            "vocab_size": 251236,
            "none_token_index": 0
        },
        "extractors": [
            {
                "domain_ontology": os.path.join(output_folder,'domain_ontology.txt'),
                "hyper-parameters": {
                    "batch_size": 50,
                    "cnn_filter_lengths": [
                        5
                    ],
                    "dropout": 0.5,
                    "entity_embedding_vector_length": 50,
                    "epoch": 30,
                    "fine-tune_epoch": 0,
                    "neighbor_distance": 3,
                    "number_of_feature_maps": 300,
                    "position_embedding_vector_length": 50,
                    "positive_weight": 5

                },
                "max_sentence_length": 301,
                "model_file": os.path.join(output_folder,'trigger.hdf'),
                "model_flags": {
                    "use_trigger": True,
                    "use_head": True,
                    "use_bio_index": False,
                    "use_lex_info": True,
                    "train_embeddings": False,
                    "early_stopping": True
                },
                "int_type": "int32",
                "model_type": "event-trigger_cnn"
            }
        ],
        "negative_trigger_words": negative_words_path,
        "train.score_file": os.path.join(output_folder,'train.score'),
        "test.score_file": os.path.join(output_folder,'test.score'),
        "save_model": True
    }
    with open(os.path.join(output_folder,'train.params.json'),'w') as fp:
        json.dump(json_param,fp,indent=4,sort_keys=True)
    with open(os.path.join(output_folder,'run_train.sh'),'w') as fp:
        fp.write("#!/bin/bash\n")
        fp.write("KERAS_BACKEND=tensorflow PYTHONPATH={}:{} {}/envs/{}/bin/python {}/nlplingo/event/train_test.py --params {} --mode train_trigger_from_file\n".format(MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,NLPLINGO_ROOT,os.path.join(output_folder,'train.params.json')))
    os.chmod(os.path.join(output_folder,'run_train.sh'),stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP)



def main(root_folder,embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,output_root):
    shrink_serifxml(root_folder)
    for root, dirs, files in os.walk(root_folder):
        if "argument.sent_spans.list" in files:
            output_folder = os.path.basename(root)
            output_folder = os.path.join(output_root,output_folder)
            make_json_param(os.path.join(root),embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV, output_folder)

if __name__ == "__main__":
    # root_folder = "/nfs/raid87/u11/users/hqiu/nlplingo_spans/wm/060619"
    # output_root = "/nfs/raid87/u11/users/hqiu/nlplingo_models/wm/060619"
    # main(root_folder, output_root)
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))
    embedding_path = c.TXT_WORDEMBEDDINGS
    negative_words_path = c.NEGATIVE_WORDS
    MY_SERIFXML_PY_PATH = c.SERIFXML_PY3_PATH
    NLPLINGO_ROOT = c.NLPLINGO_ROOT
    ANACONDA_ROOT = c.ANACONDA_ROOT
    CONDA_ENV = c.CONDA_ENV_NLPLINGO
    main(sys.argv[1],embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,sys.argv[2])