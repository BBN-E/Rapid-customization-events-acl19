import shutil,os,json,subprocess,sys,shlex

current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))

from utils.for_nlplingo.generate_nlplingo_training_json_param import main as generate_nlplingo_training_json_param

def create_symbolic_link(NLPLINGO_SPAN_OUTPUT_PATH,real_path_in_fs):
    par_dir = os.path.realpath(os.path.join(real_path_in_fs,os.path.pardir))
    os.makedirs(par_dir,exist_ok=True)
    try:
        os.symlink(NLPLINGO_SPAN_OUTPUT_PATH,real_path_in_fs)
    except:
        pass

def prepare_training_par(span_real_path_in_fs,embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,model_real_path_in_fs):
    generate_nlplingo_training_json_param(span_real_path_in_fs,embedding_path,negative_words_path,MY_SERIFXML_PY_PATH,NLPLINGO_ROOT,ANACONDA_ROOT,CONDA_ENV,model_real_path_in_fs)


def run_training(model_real_path_in_fs):
    subprocess.check_call(shlex.split("/bin/bash {}".format(os.path.join(model_real_path_in_fs,"run_train.sh"))))

if __name__ == "__main__":
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))
    embedding_path = c.TXT_WORDEMBEDDINGS
    negative_words_path = c.NEGATIVE_WORDS
    MY_SERIFXML_PY_PATH = c.SERIFXML_PY3_PATH
    NLPLINGO_ROOT = c.NLPLINGO_ROOT
    ANACONDA_ROOT = c.ANACONDA_ROOT
    CONDA_ENV = c.CONDA_ENV_NLPLINGO
    span_real_path_in_fs = sys.argv[1]
    model_real_path_in_fs = os.path.realpath(os.path.join(span_real_path_in_fs,os.pardir,"trigger_model"))

