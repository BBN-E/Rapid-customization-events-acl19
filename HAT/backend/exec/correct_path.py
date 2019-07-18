import os,sys,json,shutil
current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
from config import ReadAndConfiguration


def main(config_ins,NLPLINGO_OUTPUT_MAPPING_PATH):
    config_ins.NLPLINGO_OUTPUT_MAPPING_PATH = NLPLINGO_OUTPUT_MAPPING_PATH
    os.makedirs(config_ins.NLPLINGO_OUTPUT_MAPPING_PATH,exist_ok=True)
    for corpus in config_ins.CORPORA_FOR_DB_BOOTSTRAPING:
        old_docPath_to_new_docPath_mapping = dict()
        dst_path = os.path.join(config_ins.NLPLINGO_OUTPUT_MAPPING_PATH,'corpus')
        dst_corpus_path = os.path.join(dst_path,corpus.corpus_name)
        os.makedirs(dst_corpus_path,exist_ok=True)
        for file in os.listdir(corpus.original_corpus_folder_path):
            shutil.copy(os.path.join(corpus.original_corpus_folder_path,file),os.path.join(dst_corpus_path,file))
            # old_docPath_to_new_docPath_mapping[os.path.join(corpus.original_corpus_folder_path,file)] = os.path.join(dst_corpus_path,file)
            docId = os.path.basename(os.path.join(corpus.original_corpus_folder_path,file))
            docId = ".".join(docId.split(".")[:-1])
            old_docPath_to_new_docPath_mapping[docId] = os.path.join(dst_corpus_path,file)
        with open(os.path.join(dst_path,corpus.corpus_name+".list"),'w') as fp:
            for file in os.listdir(dst_corpus_path):
                if file.startswith(".") is False:
                    fp.write("{}\n".format(os.path.join(dst_corpus_path,file)))
        corpus.serifxml_list_path = os.path.join(dst_path,corpus.corpus_name+".list")
        corpus.cbc_m3_trigger_info = os.path.join(dst_path,"{}_trigger.ljson".format(corpus.corpus_name))
        shutil.copy(corpus.original_cbc_m3_trigger_path,corpus.cbc_m3_trigger_info)
        corpus.cbc_m3_sentence_info = os.path.join(dst_path,"{}_sentence.ljson".format(corpus.corpus_name))
        with open(corpus.original_cbc_m3_sentence_path,'r') as fp:
            with open(corpus.cbc_m3_sentence_info,'w') as wfp:
                for i in fp:
                    i = i.strip()
                    j = json.loads(i)
                    # j['docPath'] = old_docPath_to_new_docPath_mapping[j['docPath']]
                    j['docPath'] = old_docPath_to_new_docPath_mapping[j['docId']]
                    wfp.write("{}\n".format(json.dumps(j)))
    config_ins.NLPLINGO_SPAN_OUTPUT_PATH = os.path.join(config_ins.NLPLINGO_OUTPUT_MAPPING_PATH,'nlplingo','trigger')
    # config_ins.NLPLINGO_ARG_SPAN_OUTPUT_PATH = os.path.join(config_ins.NLPLINGO_OUTPUT_MAPPING_PATH,'nlplingo','argument')
    with open(os.path.join(project_root,"config_default.json"),'w') as fp:
        json.dump(config_ins.restore_to_dict(),fp,indent=4,sort_keys=True)



if __name__ == "__main__":
    ca = ReadAndConfiguration(os.path.join(project_root, "config_default.json"))
    main(ca,sys.argv[-1])