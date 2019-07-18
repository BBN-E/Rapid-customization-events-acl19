from filelock import FileLock,Timeout
import os
import subprocess
import time
import csv
from flask import json
import shutil
import sys
sys.path.append(os.getcwd())


def update_progress(config_ins,session,progress,uptime,redirect=None):
    with open(os.path.join(config_ins.FOLDER_SESSION(session),'cbc_hac_progress'),'w') as fp:
        json.dump({'progress':progress,'uptime':uptime,'redirect':redirect},fp=fp)

def main(config_ins,session,wordlist):
    lock_path = os.path.join(config_ins.FOLDER_SESSION(session),'cbc_hac.lock')
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=1):
            with open(os.path.join(config_ins.FOLDER_SESSION(session), 'cbc_hac_progress'), 'w') as fp:
                json.dump({'progress': "Scheduling", 'uptime': 0}, fp=fp)
            t = time.time()
            word_mapping = {}
            with open(wordlist,'r') as fp:
                for i in fp.readlines():
                    entry = json.loads(i)
                    key = entry['trigger'].split("_")[-1]
                    (word_mapping.setdefault(key,set())).add(entry['trigger'])
            actual_word_phrase_mapping = {}
            cnt = 0
            with open(config_ins.TXT_WORDEMBEDDINGS, 'r') as fp:
                with open(os.path.join(config_ins.FOLDER_SESSION(session), 'vocab.list'), 'w') as fp2:
                    for line in fp:
                        word = line.split(" ")[0]
                        if word in word_mapping:
                            fp2.write("{}\n".format(word))
                            actual_word_phrase_mapping[word] = list(word_mapping[word])
                            cnt = cnt + 1
            with open(os.path.join(config_ins.FOLDER_SESSION(session), 'thrown_by_embedding.txt'), 'w') as fp3:
                for key,val in word_mapping.items():
                    if actual_word_phrase_mapping.get(key) is None:
                        fp3.write("{}\n".format([word for word in val]))
            with open(os.path.join(config_ins.FOLDER_SESSION(session), 'word_phrase_mapping.json'), 'w') as fp:
                json.dump(actual_word_phrase_mapping, fp)

            # Configuration
            with open(os.path.join(config_ins.HAT_DATA_FOLDER,'ychan-pairwise.generated.par')) as fp:
                conf = fp.readlines()
            for idx,i in enumerate(conf):
                if "topK:" in i:
                    conf[idx] = "topK: {}\n".format(cnt)
                elif "outputDir:" in i:
                    conf[idx] = "outputDir: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity'))
                elif "word.vocab:" in i:
                    conf[idx] = "word.vocab: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session), 'vocab.list'))
            with open(os.path.join(config_ins.FOLDER_SESSION(session),'ychan-pairwise.generated.par'),'w') as fp:
                fp.writelines(conf)

            with open(os.path.join(config_ins.HAT_DATA_FOLDER,'hac.params')) as fp:
                conf = fp.readlines()
            for idx,i in enumerate(conf):
                if "interMemberSimilarity:" in i:
                    conf[idx] = "interMemberSimilarity: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity','sim.ALL'))
                elif "targetMembers.list:" in i:
                    conf[idx] = "targetMembers.list: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'hac.input'))
                elif "hac.output:" in i:
                    conf[idx] = "hac.output: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'hac.output'))
            with open(os.path.join(config_ins.FOLDER_SESSION(session),'hac.params'),'w') as fp:
                fp.writelines(conf)

            with open(os.path.join(config_ins.HAT_DATA_FOLDER,'ychan-clustering.generated.par')) as fp:
                conf = fp.readlines()
            for idx,i in enumerate(conf):
                if "cbc.committees:" in i:
                    conf[idx] = "cbc.committees: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'cbc','cbc.committees'))
                elif "cbc.finalClustering:" in i:
                    conf[idx] = "cbc.finalClustering: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'cbc','cbc.finalClustering'))
                elif "cbc.outputDirectory:" in i:
                    conf[idx] = "cbc.outputDirectory: {}\n".format(os.path.join(config_ins.FOLDER_SESSION(session),'cbc'))
                elif "interMemberSimilarity:" in i:
                    conf[idx] = "interMemberSimilarity: {}\n".format("{0}/sim.ALL".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity')))
                elif "targetMembers.list:" in i:
                    conf[idx] = "targetMembers.list: {}\n".format("{}/sim.ALL.vocab".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity')))
            with open(os.path.join(config_ins.FOLDER_SESSION(session),'ychan-clustering.generated.par'),'w') as fp:
                fp.writelines(conf)


            # Cleaning
            try:
                shutil.rmtree(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity'))
            except:
                pass
            try:
                shutil.rmtree(os.path.join(config_ins.FOLDER_SESSION(session),'cbc'))
            except:
                pass


            try:
                os.mkdir(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity'))
            except:
                pass

            try:
                os.mkdir(os.path.join(config_ins.FOLDER_SESSION(session),'cbc'))
            except:
                pass


            t0 = time.time() - t
            update_progress(config_ins,session,'Stage 0 finished', t0)
            subprocess.check_call([config_ins.BIN_CALCULATE_PAIRWISE_SIMILARITY,os.path.join(config_ins.FOLDER_SESSION(session),'ychan-pairwise.generated.par')])
            t1 = time.time() - t
            update_progress(config_ins,session,'Stage 1 finished', t1)
            subprocess.check_call("cat {0}/*.sim > {0}/sim.ALL".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity')),shell=True)
            t2 = time.time() - t
            update_progress(config_ins,session,'Stage 2 finished', t2)
            subprocess.check_call("cat {0}/sim.ALL | cut -d' ' -f1 > {0}/sim.ALL.col1".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity')),shell=True)
            t3 = time.time() - t
            update_progress(config_ins,session,'Stage 3 finished', t3)
            subprocess.check_call(["perl", "{}".format(os.path.join(config_ins.HAT_DATA_FOLDER,'unique.pl')) ,"{}/sim.ALL.col1".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity')), "{}/sim.ALL.vocab".format(os.path.join(config_ins.FOLDER_SESSION(session),'calculatePairwiseSimilarity'))])
            t4 = time.time() - t
            update_progress(config_ins,session,'Stage 4 finished', t4)
            subprocess.check_call([config_ins.BIN_CBC,os.path.join(config_ins.FOLDER_SESSION(session),'ychan-clustering.generated.par')])
            t5 = time.time() - t
            update_progress(config_ins,session,'Stage 5 finished', t5)
            with open(os.path.join(config_ins.FOLDER_SESSION(session),'cbc','cbc.committees'),'r') as fpr:
                with open(os.path.join(config_ins.FOLDER_SESSION(session),'hac.input'),'w') as fpw:
                    for line in fpr.readlines():
                        word = line.split(" ")[0]
                        score = line.split(" ")[1]
                        score = float(score)
                        if score < 1:
                            continue
                        fpw.write("{}\n".format(word))
            subprocess.check_call([config_ins.BIN_HAC,os.path.join(config_ins.FOLDER_SESSION(session),'hac.params')])
            t6 = time.time() - t
            update_progress(config_ins,session,'All finished', t6,'step2')

    except Timeout:
        t = os.path.getmtime(lock_path)
        update_progress(config_ins,session,'Cannot get the execution lock. Please try again later. Note: if there\'s a pending job, you cannot stop it.', -1)
        if time.time() - t > 800:
            shutil.rmtree(lock_path)
        exit(-1)
    except:
        import traceback
        traceback.print_exc()
        update_progress(config_ins,session,'Internal Error:CBC-HAC EXECUTION ERROR',-1)
        exit(-1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        exit(-1)

    config_path = sys.argv[1]
    from config import ReadAndConfiguration
    config_ins = ReadAndConfiguration(config_path)


    session = sys.argv[2]
    wordlist = sys.argv[3]
    main(config_ins,session,wordlist)
