import os,sys,json,shutil,subprocess,shlex
current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
from filelock import FileLock,Timeout
from utils.for_nlplingo.prepare_model import prepare_nlplingo_event_and_argument_model_folder
from models.nlplingo import generate_nlplingo_decoding_parameter_str
import time

class NLPLINGOWrapper(object):
    def __init__(self,anaconda_env_path,nlplingo_project_root,nlplingo_output_buffer,serifxml_list,svn_project_root=None):
        self.anaconda_env_path = anaconda_env_path
        self.nlplingo_project_root = nlplingo_project_root
        if svn_project_root is None:
            self.svn_project_root = os.path.join(project_root,'utils')
        else:
            self.svn_project_root = svn_project_root
        self.nlplingo_env = os.environ.copy()
        self.nlplingo_env["PYTHONPATH"] = "{}:{}".format(self.svn_project_root,nlplingo_project_root)
        self.nlplingo_output_buffer = nlplingo_output_buffer
        self.serifxml_list =serifxml_list

    def train_trigger_from_s3(self):
        subprocess.check_call(shlex.split("{}/bin/python {}/nlplingo/event/train_test.py --params {}/params --mode train_trigger".format(self.anaconda_env_path,self.nlplingo_project_root,self.trigger_span_s3_path)),env=self.nlplingo_env)


    def train_trigger(self):
        subprocess.check_call(shlex.split("{}/bin/python {}/nlplingo/event/train_test.py --params {}/params --mode train_trigger".format(self.anaconda_env_path,self.nlplingo_project_root,self.trigger_span_path)),env=self.nlplingo_env)

    def train_argument(self):
        subprocess.check_call(shlex.split("{}/bin/python {}/nlplingo/event/train_test.py --params {}/params --mode train_argument".format(self.anaconda_env_path,self.nlplingo_project_root,self.argument_span_path)),env=self.nlplingo_env)

    @property
    def trigger_span_s3_path(self):
        return os.path.join(self.nlplingo_output_buffer,"DummyRoot")

    @property
    def trigger_span_path(self):
        return os.path.join(self.nlplingo_output_buffer,"trigger_argument")

    @property
    def argument_span_path(self):
        return self.trigger_span_path

    @property
    def trigger_and_arugment_model_combined_folder(self):
        return os.path.join(self.nlplingo_output_buffer,"model_combined")

    @property
    def decoding_path(self):
        return os.path.join(self.nlplingo_output_buffer, "decoding")



    def combine_trigger_model_and_argument_model_s3(self):
        trigger_model_file_path = os.path.join(self.trigger_span_s3_path,"nlplingo_out")
        shutil.copy(os.path.join(trigger_model_file_path,os.path.pardir,'domain_ontology.txt'),trigger_model_file_path)
        array_of_event_arg_domain_ontology_path = [
            "/nfs/raid66/u14/users/jfaschin/runjobs/expts/nlplingo_tf/ace_09062018-role_verify_v2",
        ]
        prepare_nlplingo_event_and_argument_model_folder(trigger_model_file_path,
                                                         array_of_event_arg_domain_ontology_path, self.trigger_and_arugment_model_combined_folder)
    def combine_trigger_model_and_argument_model(self):
        trigger_model_file_path = os.path.join(self.trigger_span_path,"nlplingo_out")
        argument_model_file_path = os.path.join(self.argument_span_path,"nlplingo_out")
        shutil.copy(os.path.join(trigger_model_file_path,os.path.pardir,'domain_ontology.txt'),trigger_model_file_path)
        shutil.copy(os.path.join(argument_model_file_path,os.path.pardir,'domain_ontology.txt'),argument_model_file_path)
        array_of_event_arg_domain_ontology_path = [
            "/nfs/raid66/u14/users/jfaschin/runjobs/expts/nlplingo_tf/ace_09062018-role_verify_v2",
            argument_model_file_path
        ]
        prepare_nlplingo_event_and_argument_model_folder(trigger_model_file_path,
                                                         array_of_event_arg_domain_ontology_path, self.trigger_and_arugment_model_combined_folder)




    def decoding(self):
        os.makedirs(self.decoding_path,exist_ok=True)
        with open(os.path.join(self.decoding_path,'span_serif.list'),'w') as fp:
            for i in self.serifxml_list:
                fp.write("SERIF:{}\n".format(i))
        with open(os.path.join(self.decoding_path,'params'),'w') as fp:
            fp.write(generate_nlplingo_decoding_parameter_str(self.trigger_and_arugment_model_combined_folder,os.path.join(self.decoding_path,'span_serif.list'),self.decoding_path))
        subprocess.check_call(shlex.split(
            "{}/bin/python {}/nlplingo/event/train_test.py --params {}/params --mode decode_trigger_argument --params_json {}".format(
            self.anaconda_env_path, self.nlplingo_project_root, self.decoding_path,os.path.join(self.trigger_and_arugment_model_combined_folder,'decoding.json'))), env=self.nlplingo_env)

def update_progress(c,session_id,progress,uptime,error_msg=None,redirect=None):
    with open(os.path.join(c.FOLDER_SESSION(session_id), 'nlplingo', 'progress.json'), 'w') as fp:
        json.dump({"progress":progress,"uptime":uptime,"error_msg":error_msg,"redirect":redirect},fp)

def train_model_from_s3(c,session_id):
    lock_path = os.path.join(c.FOLDER_SESSION(session_id),'trigger_argument.lock')
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=1):
            import pymongo
            mongo_db = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
            from utils.for_nlplingo.parse_prediction_json import parse_prediction_json
            from data_access_layer.trigger_annotation.mongodb import TriggerAnnotationMongoDBAccessLayer
            from data_access_layer.common.nlplingo import nlplingo_serializer_flatten_flavor
            anaconda_env_path = c.anaconda_env_path
            nlplingo_project_root = c.nlplingo_project_root
            nlplingo_training_root = os.path.join(c.FOLDER_SESSION(session_id), 'nlplingo')
            os.makedirs(nlplingo_training_root, exist_ok=True)
            decoding_serifxml_list = list()
            # enabled_corpora = set(c.CORPORA_FOR_UI_BOOTSTRAPING)
            with open(os.path.join(c.FOLDER_SESSION(session_id), 'working_corpora.json'), 'r') as fp:
                enabled_corpora = json.load(fp)
            start_moment = time.time()

            for corpus_name in enabled_corpora:
                for en in mongo_db[c.DOCID_TABLE_NAME].find({'corpusName': corpus_name}):
                    decoding_serifxml_list.append(en['docPath'])

            decoding_serifxml_list = decoding_serifxml_list[:100]

            nlplingo_wrapper = NLPLINGOWrapper(anaconda_env_path, nlplingo_project_root, nlplingo_training_root,
                                               decoding_serifxml_list)
            dao = TriggerAnnotationMongoDBAccessLayer()
            doc_id_table_name = c.DOCID_TABLE_NAME
            annotations = dao.dump_annotation_out(mongo_db, session_id, doc_id_table_name)

            nlplingo_serializer_flatten_flavor(annotations, nlplingo_wrapper.trigger_span_s3_path)

            update_progress(c, session_id, "Start trigger training", time.time() - start_moment)
            nlplingo_wrapper.train_trigger_from_s3()
            nlplingo_wrapper.combine_trigger_model_and_argument_model_s3()

            update_progress(c, session_id, "Start decoding", time.time() - start_moment)
            nlplingo_wrapper.decoding()
            with open(os.path.join(nlplingo_wrapper.decoding_path, 'prediction.json'), 'r') as fp:
                j = json.load(fp)
            from data_access_layer.argument_annotation.mongodb import ArgumentAnnotationMongoDBAccessLayer
            update_progress(c, session_id, "Dump decoding result into mongo", time.time() - start_moment)
            candidates = parse_prediction_json(j, decoding_serifxml_list)
            dao = ArgumentAnnotationMongoDBAccessLayer()
            dao.dump_candidates(candidates, mongo_db, session_id, c.DOCID_TABLE_NAME)
            update_progress(c, session_id, "Finished", time.time() - start_moment, redirect="step4")
    except Timeout:
        t = os.path.getmtime(lock_path)
        update_progress(c,session_id,'Cannot get the execution lock. Please try again later. Note: if there\'s a pending job, you cannot stop it.', -1)
        if time.time() - t > 800:
            shutil.rmtree(lock_path)
        exit(-1)
    except:
        import traceback
        traceback.print_exc()
        update_progress(c,session_id,'Internal Error:EXECUTION ERROR',-1)
        exit(-1)

def train_model_from_s4(c,session_id):
    lock_path = os.path.join(c.FOLDER_SESSION(session_id),'trigger_argument.lock')
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=1):
            import pymongo
            mongo_db = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
            from utils.for_nlplingo.parse_prediction_json import parse_prediction_json
            from data_access_layer.trigger_annotation.mongodb import TriggerAnnotationMongoDBAccessLayer
            from data_access_layer.argument_annotation.mongodb import ArgumentAnnotationMongoDBAccessLayer
            from data_access_layer.common.nlplingo import nlplingo_serializer_flatten_flavor
            anaconda_env_path = c.anaconda_env_path
            nlplingo_project_root = c.nlplingo_project_root
            nlplingo_training_root = os.path.join(c.FOLDER_SESSION(session_id), 'nlplingo')
            os.makedirs(nlplingo_training_root, exist_ok=True)
            decoding_serifxml_list = list()
            # enabled_corpora = set(c.CORPORA_FOR_UI_BOOTSTRAPING)
            with open(os.path.join(c.FOLDER_SESSION(session_id), 'working_corpora.json'), 'r') as fp:
                enabled_corpora = json.load(fp)
            start_moment = time.time()



            for corpus_name in enabled_corpora:
                for en in mongo_db[c.DOCID_TABLE_NAME].find({'corpusName': corpus_name}):
                    decoding_serifxml_list.append(en['docPath'])

            decoding_serifxml_list = decoding_serifxml_list[:100]

            nlplingo_wrapper = NLPLINGOWrapper(anaconda_env_path, nlplingo_project_root, nlplingo_training_root,
                                               decoding_serifxml_list)
            doc_id_table_name = c.DOCID_TABLE_NAME

            dao = TriggerAnnotationMongoDBAccessLayer()
            annotations_from_s3 = dao.dump_annotation_out(mongo_db,session_id,doc_id_table_name)

            dao = ArgumentAnnotationMongoDBAccessLayer()
            annotations_from_s4 = dao.dump_annotation_out(mongo_db, session_id, doc_id_table_name)

            annotations = dict()
            # Merge s3 annotation with s4

            for trigger_info,argument_set in annotations_from_s3.items():
                annotations.setdefault(trigger_info,set()).update(argument_set)
            for trigger_info,argument_set in annotations_from_s4.items():
                annotations.setdefault(trigger_info,set()).update(argument_set)

            nlplingo_serializer_flatten_flavor(annotations, nlplingo_wrapper.trigger_span_path)

            update_progress(c, session_id, "Start trigger training", time.time() - start_moment)
            nlplingo_wrapper.train_trigger()
            update_progress(c, session_id, "Start argument training", time.time() - start_moment)
            nlplingo_wrapper.train_argument()
            nlplingo_wrapper.combine_trigger_model_and_argument_model()
            update_progress(c, session_id, "Start decoding", time.time() - start_moment)
            nlplingo_wrapper.decoding()
            with open(os.path.join(nlplingo_wrapper.decoding_path, 'prediction.json'), 'r') as fp:
                j = json.load(fp)
            from data_access_layer.argument_annotation.mongodb import ArgumentAnnotationMongoDBAccessLayer
            update_progress(c, session_id, "Dump decoding result into mongo", time.time() - start_moment)
            candidates = parse_prediction_json(j, decoding_serifxml_list)
            dao = ArgumentAnnotationMongoDBAccessLayer()
            dao.dump_candidates(candidates, mongo_db, session_id, c.DOCID_TABLE_NAME)
            update_progress(c, session_id, "Finished", time.time() - start_moment, redirect="step4")
    except Timeout:
        t = os.path.getmtime(lock_path)
        update_progress(c,session_id,'Cannot get the execution lock. Please try again later. Note: if there\'s a pending job, you cannot stop it.', -1)
        if time.time() - t > 800:
            shutil.rmtree(lock_path)
        exit(-1)
    except:
        import traceback
        traceback.print_exc()
        update_progress(c,session_id,'Internal Error:EXECUTION ERROR',-1)
        exit(-1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--config_path")
    parser.add_argument("--session_id")
    args = parser.parse_args()
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(args.config_path)
    if args.mode == "train_model_from_s3":
        train_model_from_s3(c,args.session_id)
    else:
        train_model_from_s4(c,args.session_id)