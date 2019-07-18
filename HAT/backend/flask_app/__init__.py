import os
import sys

current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))

from flask import Flask,request,jsonify,send_from_directory,g,current_app
from flask_cors import CORS
import subprocess
import bisect
import operator
import shutil
from flask import json
import sys
import re
import time
sys.path.append(os.getcwd())
from config import ReadAndConfiguration
from werkzeug.utils import secure_filename
import datetime
import pymongo
import shlex
import random
import numpy as np
from scipy.spatial import distance


from nlp.common import Marking
from data_access_layer.argument_annotation.mongodb import ArgumentAnnotationMongoDBAccessLayer
from data_access_layer.trigger_annotation.mongodb import TriggerAnnotationMongoDBAccessLayer
from models.frontend import TriggerArgumentInstance
from nlp.EventMention import EventArgumentMentionIdentifierTokenIdxBase,EventMentionInstanceIdentifierTokenIdxBase
from utils.lemmaizer import Lemmaizer
from utils.json_encoder import ComplexEncoder

flaskapp = Flask(__name__)
flaskapp.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
flaskapp.json_encoder = ComplexEncoder
CORS(flaskapp)
c = ReadAndConfiguration(os.path.join(project_root, "config_default.json"))
print("We're running {}".format(c.__class__))
lemmaizer = Lemmaizer(c.LEMMA_FILE_PATH)
mongo_db = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)


@flaskapp.errorhandler(404)
def not_found(error):
    return jsonify({'text': 'Not a valid endpoint'}), 404


@flaskapp.errorhandler(403)
def forbidden(error):
    return jsonify({'text': 'Credentials needed'}), 403


@flaskapp.errorhandler(410)
def gone(error):
    return jsonify({'text': 'The endpoint is abandoned'}), 410


@flaskapp.errorhandler(405)
def gone(error):
    return jsonify({'text': 'You should not do that.'}), 405


@flaskapp.errorhandler(Exception)
def internal_server_error(error):
    from traceback import format_exc, print_exc
    current_app.logger.error(json.dumps({'msg': format_exc(), 'msg_type': "500"}))
    if flaskapp.config.get('DEBUG') is True:
        print_exc()
    return jsonify({'text': format_exc()}), 500



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@flaskapp.before_first_request
def cleanup():
    try:
        os.makedirs(c.BACKEND_RUNTIME_FOLDER)
    except:
        pass


@flaskapp.before_request
def bind_session_id():
    session = request.args.get('session',"dummy")
    session = secure_filename(session)
    en = os.path.exists(os.path.join(c.FOLDER_SESSION(session)))
    if en is False:
        try:
            os.makedirs(os.path.join(c.FOLDER_SESSION(session)))
        except:
            pass
        lock_path = os.path.join(c.FOLDER_SESSION(session), 'cbc_hac.lock')
        try:
            shutil.rmtree(lock_path)
        except:
            pass
        with open(os.path.join(c.FOLDER_SESSION(session), 'cbc_hac_progress'), 'w') as fp:
            json.dump({'progress': "Not running", 'uptime': 0}, fp=fp)
    try:
        os.makedirs(os.path.join(c.FOLDER_SESSION(session)))
    except:
        pass
    g.session = session

@flaskapp.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    r.headers['Last-Modified'] = datetime.datetime.now()
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "-1"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



@flaskapp.route('/exception_tester')
def throw_an_exception():
    raise Exception("Haha you got an exception")



@flaskapp.route('/s1page/<path:path>')
def send_statics(path):
    return send_from_directory(directory=os.path.join(c.FOLDER_SESSION(g.session),"s1page"),filename=path,mimetype="application/json")

def reset_cbc_wordlist():
    # Only work aftye
    counter_dict_word = {}
    counter_dict_phrase = {}
    cnt = 0
    with open(os.path.join(c.FOLDER_SESSION(g.session), 'cbc-m3-trigger.tsv')) as fp:
        for i in fp.readlines():
            entry = json.loads(i)
            entry['sentenceId'] = str((entry['sentenceId']['key'], entry['sentenceId']['value']))
            if entry['triggerPosTag'] == "NP":
                cur_cnt = counter_dict_phrase.get((entry['trigger'], entry['triggerPosTag']), 0)
                counter_dict_phrase[(entry['trigger'], entry['triggerPosTag'])] = cur_cnt + 1
            else:
                cur_cnt = counter_dict_word.get((entry['trigger'], entry['triggerPosTag']), 0)
                counter_dict_word[(entry['trigger'], entry['triggerPosTag'])] = cur_cnt + 1
            cnt += 1
            if cnt % 10000 == 0:
                print("Examined {} lines of triggers.".format(cnt))
    out_word = [{'trigger': key[0], 'trigger_postag': key[1], 'cnt': val, 'id': key[0] + '.' + key[1]} for key, val in
                list(counter_dict_word.items())]
    out_phrase = [{'trigger': key[0], 'trigger_postag': key[1], 'cnt': val, 'id': key[0] + '.' + key[1]} for key, val in
                  list(counter_dict_phrase.items())]
    before_whitelist_out = out_word + out_phrase
    before_whitelist_out = sorted(before_whitelist_out, key=lambda d: d['cnt'], reverse=True)
    whitelist_words_and_phrases = set()
    for i in before_whitelist_out:
        whitelist_words_and_phrases.add(i['trigger'])
    out = list()
    for i in before_whitelist_out:
        if i.get("trigger") in whitelist_words_and_phrases:
            out.append(i)
    with open(os.path.join(c.FOLDER_SESSION(g.session), 's2_wordlist_from_s1.txt'), 'w') as fp:
        for i in out:
            fp.write("{}\n".format(json.dumps(i)))
    return out


@flaskapp.route('/s1/init',methods=['POST'])
def init_session():
    out = reset_cbc_wordlist()
    MakePagination(out, 15, 's1page')
    return send_from_directory(os.path.join(c.FOLDER_SESSION(g.session),'s1page'),"0.json")

@flaskapp.route('/s1/reset',methods=['POST'])
def reset_cbc_word_list():
    out = reset_cbc_wordlist()
    MakePagination(out, 15, 's1page')
    return send_from_directory(os.path.join(c.FOLDER_SESSION(g.session),'s1page'),"0.json")

def try_or_set(val,t,defau):
    try:
        return t(val)
    except:
        return t(defau)

def UserWordFilter(stopword,word_threshold,NP_threshold):
    out_word = list()
    out_phrase = list()
    with open(os.path.join(c.FOLDER_SESSION(g.session),'s2_wordlist_from_s1.txt'),'r') as fp:
        for i in fp.readlines():
            current_item = json.loads(i)
            if current_item.get("trigger_postag") == "NP":
                out_phrase.append(current_item)
            else:
                out_word.append(current_item)
    out_word = sorted(out_word,key=lambda d:d['cnt'],reverse=True)
    filter_out_word = [item for item in out_word if (item['cnt'] < word_threshold) or ((item.get('trigger'),item.get('trigger_postag')) in stopword)]
    out_word = [item for item in out_word if (item['cnt'] >= word_threshold) and ((item.get('trigger'),item.get('trigger_postag')) not in stopword)]
    out_phrase = sorted(out_phrase,key=lambda d:d['cnt'],reverse=True)
    filter_out_phrase = [item for item in out_phrase if (item['cnt'] < NP_threshold) or ((item.get('trigger'),item.get('trigger_postag')) in stopword)]
    out_phrase = [item for item in out_phrase if (item['cnt'] >= NP_threshold) and ((item.get('trigger'),item.get('trigger_postag')) not in stopword)]
    out = out_word + out_phrase
    with open(os.path.join(c.FOLDER_SESSION(g.session),'s2_wordlist_from_s1.txt'),'w') as fp:
        for i in out:
            fp.write("{}\n".format(json.dumps(i)))
    return out_word,out_phrase,filter_out_word,filter_out_phrase

def MakePagination(data,n,folder_name):
    # Pagination
    try:
        shutil.rmtree(os.path.join(c.FOLDER_SESSION(g.session), folder_name))
    except:
        pass
    os.mkdir(os.path.join(c.FOLDER_SESSION(g.session), folder_name))
    total = len(data)
    out = [data[i:i + n] for i in range(0, len(data), n)]
    for idx, i in enumerate(out):
        with open(os.path.join(c.FOLDER_SESSION(g.session), folder_name, "{}.json".format(idx)), 'w') as fp:
            ret = {}
            ret["links"] = {}
            ret["links"]["pagination"] = {
                "total": total,
                "per_page": n,
                "current_page": idx + 1,
                "last_page": len(out),
                "next_page_url": "/{}/{}.json?session={}".format(folder_name,min(idx + 1, len(out) - 1), g.session),
                "prev_page_url": "/{}/{}.json?session={}".format(folder_name,max(idx - 1, 0), g.session),
                "from": idx * n + 1,
                "to": (idx + 1) * n,
            }
            ret["data"] = i
            json.dump(ret, fp)


@flaskapp.route('/s1',methods=['POST'])
def step1():
    stopword = request.json.get('stopword',[])
    stopword = [(".".join(item.split('.')[0:-1]),item.split('.')[-1]) for item in stopword]
    stopword = set(stopword)
    frequency_word_filter = request.json.get('frequency_word',-1)
    frequency_phrase_filter = request.json.get('frequency_phrase',-1)
    frequency_word_filter = try_or_set(frequency_word_filter,int,-1)
    frequency_phrase_filter = try_or_set(frequency_phrase_filter,int,-1)
    out_word, out_phrase, filter_out_word, filter_out_phrase = UserWordFilter(stopword,frequency_word_filter,frequency_phrase_filter)
    out = out_word + out_phrase
    out = sorted(out,key=lambda d:d['cnt'],reverse=True)
    MakePagination(out,15,'s1page')
    return send_from_directory(os.path.join(c.FOLDER_SESSION(g.session),'s1page'),"0.json")

@flaskapp.route('/s2/fromstep1',methods=['POST'])
def step2_fromstep1():
    stopword = request.json.get('stopword', [])
    stopword = [(".".join(item.split('.')[0:-1]),item.split('.')[-1]) for item in stopword]
    stopword = set(stopword)
    frequency_word_filter = request.json.get('frequency_word',-1)
    frequency_phrase_filter = request.json.get('frequency_phrase',-1)
    frequency_word_filter = try_or_set(frequency_word_filter,int,-1)
    frequency_phrase_filter = try_or_set(frequency_phrase_filter,int,-1)

    if len(stopword) > 0 or frequency_word_filter > 0 or frequency_phrase_filter > 0:
        out_word, out_phrase, filter_out_word, filter_out_phrase = UserWordFilter(stopword, frequency_word_filter,
                                                                                  frequency_phrase_filter)
    subprocess.Popen(["python",os.path.join(os.getcwd(), 'external_executable_wrapper/cbc_hac_wrapper.py'),c.CONFIG_FILE_PATH,g.session,os.path.join(c.FOLDER_SESSION(g.session),'s2_wordlist_from_s1.txt')])
    return "OK"

@flaskapp.route('/s2/fromfile',methods=['POST'])
def step2_fromfile():
    wordlist = request.files.get('wordlist')
    words = wordlist.read().decode('utf-8')
    words = words.replace('\r','')
    words = words.split('\n')
    words = list(filter(lambda x:len(x)>0,words))
    words = set(words)
    with open(os.path.join(c.FOLDER_SESSION(g.session),'s2_wordlist_from_user.txt'),'w') as fp:
        for i in words:
            fp.write("{}\n".format(json.dumps({"trigger":i})))
    if os.path.isfile(os.path.join(c.FOLDER_SESSION(g.session),'sim.ALL')) is False:
        try:
            shutil.copy(os.path.join(c.FOLDER_SESSION(g.session),'calculatePairwiseSimilarity','sim.ALL'),os.path.join(c.FOLDER_SESSION(g.session),'sim.ALL'))
        except:
            pass
    subprocess.Popen(["python", os.path.join(os.getcwd(), 'external_executable_wrapper/cbc_hac_wrapper.py'), g.session,os.path.join(c.FOLDER_SESSION(g.session),'s2_wordlist_from_user.txt')])
    return "OK"


@flaskapp.route('/s2/2')
def step2_2():
    return send_from_directory(directory=os.path.join(c.FOLDER_SESSION(g.session)), filename='cbc_hac_progress',
                               mimetype="application/json")


def shrink_tree_to_limited_height(orig_path_strs,n_layers):
    pattern = re.compile(r'c\d+$')
    resolved_tree = list()
    appeared_path = set()
    for i in orig_path_strs:
        i = i.strip()
        i = i.split(".")
        cur_buf = i[:n_layers]
        for j in range(n_layers,len(i)):
            if not pattern.match(i[j]):
                cur_buf.append(i[j])
        solved_str = ".".join(cur_buf)
        if solved_str not in appeared_path:
            appeared_path.add(solved_str)
            resolved_tree.append(solved_str)
    return resolved_tree

def my_fake_lemmaizer(token):
    token = token.lower()
    token = lemmaizer.get_lemma(token,False)
    # if token.endswith("es"):
    #     token = token[:-2]
    # elif token.endswith("s"):
    #     token = token[:-1]
    # elif token.endswith("ed"):
    #     token = token[:-1]
    # elif token.endswith("ing"):
    #     token = token[:-3]
    # elif token.endswith("."):
    #     token = token[:-1]
    return token

@flaskapp.route('/s2/3')
def step2_3():
    ret = {}
    ret['csv'] = []
    cnt = 0
    cbc_rep_to_clusters = {}
    cbc_theshold = 5.5

    cbc_committee_members = set()
    blacklisted_emb_ids = set()
    with open(os.path.join(c.FOLDER_SESSION(g.session),'blacklist.json')) as fp:
        blacklisted_embs = json.load(fp)
    for i in blacklisted_embs:
        if "bert_emb_idx" in i.keys():
            blacklisted_emb_ids.add(i["bert_emb_idx"])

    with open(os.path.join(c.FOLDER_SESSION(g.session),'cbc.committees'),'r') as fp:
        for i in fp.readlines():
            i = i.replace('\n','')
            d = i.split(" ")
            score = float(d[1])
            if score > cbc_theshold:
                cbc_committee_members.add(d[0])
                cbc_rep_to_clusters.setdefault(d[0],set()).update([d[item] for item in range(2,len(d))])
    with open(os.path.join(c.FOLDER_SESSION(g.session),'cbc.finalClustering'),'r') as fp:
        for i in fp.readlines():
            i = i.replace('\n','')
            d = i.split(" ")
            score = float(d[1])
            if score > cbc_theshold:
                cbc_committee_members.add(d[0])
                cbc_rep_to_clusters.setdefault(d[0],set()).update([d[item] for item in range(2,len(d))])

    cbc_rep_to_clusters = {k:list(v) for k,v in cbc_rep_to_clusters.items()}

    with open(os.path.join(c.FOLDER_SESSION(g.session),'cbc.representative'),'w') as fp:
        for i in cbc_committee_members:
            fp.write("{}\n".format(i))
    subprocess.check_call([c.BIN_HAC,os.path.join(c.FOLDER_SESSION(g.session),'hac.par')])

    pattern = re.compile(r'c\d+$')

    bert_emb_idx_to_en = dict()
    with open(os.path.join(c.FOLDER_SESSION(g.session),'alignment_shrink.ljson')) as fp:
        for i in fp:
            i = i.strip()
            j = json.loads(i)
            bert_emb_idx_to_en['BERTEMBIDX_{}'.format(j['original_idx'])] = j



    with open(os.path.join(c.FOLDER_SESSION(g.session),'hac.out')) as fp:
        # shrinked_tree = shrink_tree_to_limited_height(fp.readlines(),n_layers=22)
        shrinked_tree = [i.strip() for i in fp.readlines()]

    node_id_to_node = dict()
    root_node = None
    for i in shrinked_tree:
        if "." not in i:
            parent_id = i
            parent_node = {'id':i,'value':i,'bert_emb_idx':None,'children':[]}
            node_id_to_node[parent_id] = parent_node
            root_node = parent_node
            continue
        else:
            parent_id = ".".join(i.split(".")[:-1])
            parent_node = node_id_to_node[parent_id]
        if not pattern.match(i.split('.')[-1]):
            bert_id = i.split('.')[-1]
            tokens = bert_emb_idx_to_en[bert_id]['orig_sentence'].split(" ")
            escaped_tokens = list()
            for idx,token in enumerate(tokens):
                added_str = token
                if idx == bert_emb_idx_to_en[bert_id]['trig_idx_start']:
                    added_str = "<strong><font color=\"#33A1DE\"><span class=\"slot0\">" + added_str
                if idx == bert_emb_idx_to_en[bert_id]['trig_idx_end']:
                    added_str = added_str + "</span></font></strong>"
                escaped_tokens.append(added_str)
            current_node = {'id':i,'value':my_fake_lemmaizer(bert_emb_idx_to_en[bert_id]['anchor_head_text']),'bert_emb_idx':bert_id,'sentence':" ".join(escaped_tokens),'children':[]}
            parent_node['children'].append(current_node)
            node_id_to_node[i] = parent_node
            # Handle cbc members

            existed_head_set = set()
            MAX_CBC_RESULT_PER_CLUSTER = 10
            for cbc_child in cbc_rep_to_clusters.get(bert_id,list()):
                if my_fake_lemmaizer(bert_emb_idx_to_en[cbc_child]['anchor_head_text']) in existed_head_set or MAX_CBC_RESULT_PER_CLUSTER < 1:
                    continue
                tokens = bert_emb_idx_to_en[cbc_child]['orig_sentence'].split(" ")
                escaped_tokens = list()
                for idx,token in enumerate(tokens):
                    added_str = token
                    if idx == bert_emb_idx_to_en[cbc_child]['trig_idx_start']:
                        added_str = "<strong><font color=\"#33A1DE\"><span class=\"slot0\">" + added_str
                    if idx == bert_emb_idx_to_en[cbc_child]['trig_idx_end']:
                        added_str = added_str + "</span></font></strong>"
                    escaped_tokens.append(added_str)
                cbc_current_node = {'id':i+".{}".format(cbc_child),'value':my_fake_lemmaizer(bert_emb_idx_to_en[cbc_child]['anchor_head_text'].lower()),'bert_emb_idx':cbc_child,'sentence':" ".join(escaped_tokens),'children':[]}
                current_node['children'].append(cbc_current_node)
                node_id_to_node[i+".{}"] = cbc_current_node
                existed_head_set.add(my_fake_lemmaizer(bert_emb_idx_to_en[cbc_child]['anchor_head_text']))
                MAX_CBC_RESULT_PER_CLUSTER -= 1

        else:
            current_node = {'id':i,'value':i.split(".")[-1],'bert_emb_id':None,'children':[]}
            parent_node['children'].append(current_node)
            node_id_to_node[i] = current_node
    ret['tree'] = root_node
    ret['links'] = list()
    for node_id,node in node_id_to_node.items():
        for child in node['children']:
            ret['links'].append({'source':node['id'],'target':child['id']})
    if os.path.exists(os.path.join(c.FOLDER_SESSION(g.session),'s3clusters.json')):
        with open(os.path.join(c.FOLDER_SESSION(g.session),'s3clusters.json'),'r') as fp:
            j = json.load(fp)
            clusters = j['clusters']
        for cluster in clusters:
            cur = []
            for trigger in cluster['children']:
                cur.append(trigger['key'])
            ret.setdefault('clusters', []).append(cur)
    ret.setdefault('clusters', [])
    return jsonify(ret)

@flaskapp.route("/s2/blacklist_nodes",methods=['POST'])
def blacklisted_node():
    blacklisted_emb_ids = request.json.get("csvs")
    with open(os.path.join(c.FOLDER_SESSION(g.session),'blacklist.json')) as fp:
        blacklisted_embs = json.load(fp)
    blacklisted_embs.extend(blacklisted_emb_ids)
    with open(os.path.join(c.FOLDER_SESSION(g.session),'blacklist.json'),'w') as fp:
        fp.write("[\n")
        fp.write(",\n".join([json.dumps(i) for i in blacklisted_embs]))
        fp.write("\n]")
    return "OK"

@flaskapp.route("/s2/get_similar_eventtypes",methods=['POST'])
def get_similar_eventtypes():
    event_mention_emb_ids = request.json.get('event_mention_emb_ids')
    care_emd_idx = set()
    # Step 1, fetch embeddings
    for event_mention_emb in event_mention_emb_ids:
        care_emd_idx.add(int(event_mention_emb.replace("BERTEMBIDX_","")))

    if len(care_emd_idx) < 1:
        return jsonify({"ret":{}})

    # Step 2, calculate centroid
    current_vec_sum = None
    current_cnt = 0
    with open(os.path.join(c.FOLDER_SESSION(g.session),"test.json_shrink.jsonl")) as fp:
        for i in fp:
            i = i.strip()
            j = json.loads(i)
            if j['original_idx'] in care_emd_idx:
                vec = np.array(j['layers'][0]['values'])
                if current_vec_sum is None:
                    current_vec_sum = np.zeros(vec.shape[0])
                current_vec_sum += vec
                current_cnt += 1
    event_mention_centroid = np.true_divide(current_vec_sum,current_cnt)
    # Step 3, calculate consine simiarity
    with open(os.path.join(c.FOLDER_SESSION(g.session),'type_to_vec.jsonl')) as fp:
        type_to_centroid = json.load(fp)
        type_to_centroid = {k:np.array(v) for k,v in type_to_centroid.items()}
    type_to_similarity = dict()
    for event_type,annotated_centroid in type_to_centroid.items():
        type_to_similarity[event_type] = distance.cosine(event_mention_centroid,annotated_centroid)
    return jsonify({"ret":type_to_similarity})


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

def convertOffsetTaggedStringToSplitList(s,minIdx):
    partial = s.split(' ')
    partial_acc = list(accumulate([len(x) for x in partial]))
    for idx, i in enumerate(partial_acc):
        partial_acc[idx] = idx + partial_acc[idx]
    if minIdx < 0:
        idx = minIdx
    else:
        idx = bisect.bisect_left(partial_acc, minIdx)
    return partial,idx


@flaskapp.route('/s2/4', methods=['POST'])
def SaveDraftClustersFromStep2():
    clusters = request.json.get('clusters', [])
    with open(os.path.join(c.FOLDER_SESSION(g.session), 's2_clusters_to_s3'), 'w') as fp:
        json.dump(clusters, fp)
    # subprocess.Popen(["python", os.path.join(os.getcwd(), 'assemble_cluster/run.py'), g.session])
    with open(os.path.join(c.FOLDER_SESSION(g.session), 'cbc_hac_progress'), 'w') as fp:
        json.dump({'progress': 'Assembling, Please wait'}, fp=fp)
    return "OK"


@flaskapp.route('/v2/s2/4',methods=['POST'])
def ClustersAssembleFromStep2():
    clusters_from_step2 = request.files.get('clusters')
    clusters_from_step2 = clusters_from_step2.read().decode('utf-8')
    clusters_from_step2 = json.loads(clusters_from_step2)
    with open(os.path.join(c.FOLDER_SESSION(g.session), 's2_clusters_to_s3'), 'w') as fp:
        json.dump(clusters_from_step2,fp)
    subprocess.Popen(["python", os.path.join(os.getcwd(), 'assemble_cluster/run.py'), g.session])
    with open(os.path.join(c.FOLDER_SESSION(g.session), 'cbc_hac_progress'), 'w') as fp:
        json.dump({'progress': 'Assembling, Please wait','redirect':'pending'}, fp=fp)
    return jsonify({'progress': 'Assembling, Please wait','redirect':'pending'})

def MarkTriggerInstances(s3clusters):
    dao = TriggerAnnotationMongoDBAccessLayer()
    for cluster in s3clusters:
        event_type_name = cluster['key']
        for trigger in cluster['children']:
            for sentence in trigger['children']:
                # TODO : Remove try catch block,it's a frontend bug
                try:
                    trigger_argument_obj = TriggerArgumentInstance.fromJSON(sentence)
                    event_id = trigger_argument_obj.instance_identifier_token_idx_base
                    if sentence.get('aux',dict()).get('touched',False) is True:
                        marking = sentence['aux']['marking']
                        dao.modify_annotation(mongo_db, g.session, c.DOCID_TABLE_NAME, lemmaizer, event_id, event_type_name,
                                              Marking(marking))
                    else:
                        dao.modify_annotation(mongo_db,g.session,c.DOCID_TABLE_NAME,lemmaizer,event_id,event_type_name,Marking.POSITIVE)
                except:
                    import traceback
                    print("WARNING: You're dropping annotation")
                    traceback.print_exc()


@flaskapp.route('/s3/data',methods=['GET','POST','DELETE'])
def ClustersDataAccessObJect():
    if request.method=="GET":
        if os.path.exists(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json')) is False:
            with open(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json'),'w') as fp:
                json.dump({'clusters':[]},fp)
        with open(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json'), 'r') as fp:
            ret = json.load(fp)
            return jsonify(ret)
    elif request.method=="POST":
        clusters = request.json.get('clusters', [])
        with open(os.path.join(c.FOLDER_SESSION(g.session),'s3clusters_{}.json'.format(time.time())),'w') as fp:
            json.dump({'clusters': clusters}, fp)
        MarkTriggerInstances(clusters)
        for cluster in clusters:
            for trigger in cluster.get('children',list()):
                trigger['children'] = list()
                trigger['aux']['blacklist'] = list()
        with open(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json'), 'w') as fp:
            json.dump({'clusters': clusters}, fp)
        return "OK"
    elif request.method=="DELETE":
        try:
            os.remove(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json'))
        except:
            pass
        return "OK"


@flaskapp.route('/s3/get_annotated_sentence',methods=['POST'])
def AnnotatedSentenceGetterFromUser():
    triggers = request.json.get('triggers',[])
    event_type = request.json.get('eventType')
    try:
        with open(os.path.join(c.FOLDER_SESSION(g.session),'ui_event_type_to_ontology_event_type_mapping.json'),'r') as fp:
            event_type_mapping = json.load(fp)
    except IOError as e:
        import traceback
        traceback.print_exc()
        event_type_mapping = {}
    dao = TriggerAnnotationMongoDBAccessLayer()
    ret = dao.annotated_sentence_getter(mongo_db,event_type,event_type_mapping.get(event_type,event_type),g.session,triggers,c.DOCID_TABLE_NAME,c.ADJUDICATED_SENTENCE_TABLE_NAME,c.ANNOTATED_INSTANCE_TABLE_NAME)
    return jsonify(ret)


@flaskapp.route('/s3/query_unannotated_sentence',methods=['POST'])
def NextGenUnannotatedSentenceGetterFromUser():
    triggers = request.json.get('triggers',[])
    limit = request.json.get('limit',c.NUM_DEFAULT_NUM_OF_SENTENCE_IN_TRIGGER)
    limit = int(limit)
    event_type = request.json.get('eventType',None)
    dao = TriggerAnnotationMongoDBAccessLayer()
    with open(os.path.join(c.FOLDER_SESSION(g.session), 'working_corpora.json'), 'r') as fp:
        corpora = json.load(fp)
    trigger_key_to_sentence_mapping = dict()
    for corpus_name in corpora:
        en = dao.unannotated_sentence_getter(mongo_db,g.session,triggers,limit,corpus_name,event_type)
        for i in en:
            key = i['key']
            if key in trigger_key_to_sentence_mapping:
                trigger_key_to_sentence_mapping[key]['aux']['blacklist'].extend(i['aux']['blacklist'])
                trigger_key_to_sentence_mapping[key]['children'].extend(i['children'])
            else:
                trigger_key_to_sentence_mapping[key] = i
    buf = list()
    for _, i in trigger_key_to_sentence_mapping.items():
        buf.append(i)
    ret = list()
    for trigger_entry in buf:
        random.shuffle(trigger_entry['children'])
        trigger_entry['children'] = trigger_entry['children'][:limit]
        ret.append(trigger_entry)
    return jsonify(ret)

@flaskapp.route('/s3/mark_sentence',methods=['POST'])
def MarkTriggerInstanceAsBad():
    trigger_instance = request.json.get('trigger_instance')
    event_type_name = request.json.get('event_type_name')
    decision = request.json.get('marking')
    dao = TriggerAnnotationMongoDBAccessLayer()
    event_id = EventMentionInstanceIdentifierTokenIdxBase.fromJSON(trigger_instance)
    dao.modify_annotation(mongo_db,g.session,c.DOCID_TABLE_NAME,lemmaizer,event_id,event_type_name,decision)
    return jsonify({'msg':"OK"}),200

@flaskapp.route('/v2/s2/similarcluster',methods=['POST'])
def getSimilarClusterInWords():
    '''
    Attention!!!
    This function need a table called pairwise_m5m6m9 support at database
    :return:
    '''
    from find_similar import find_similar
    import numpy as np
    exist_triggers = request.json.get('triggers',[])
    threshold = request.json.get('threshold',0.0)
    threshold = float(threshold)
    cutoff = request.json.get('cutoff',10)
    cutoff = int(cutoff) + 1
    mtx = np.load(c.JOSHUA_FIND_SIMILAR_MATRIX_NPZ)
    with open(c.JOSHUA_FIND_SIMILAR_MATRIX_INDEX,'r') as fp:
        idx_map = json.load(fp)
    my_db = find_similar.FakeDB(idx_map,mtx)
    trigger_words = set([trigger.split("_")[-1] for trigger in exist_triggers])
    out = find_similar.getSimilarClusterInWords(trigger_words, threshold ,cutoff, my_db)
    out = list(map(lambda x:{'key':x,'aux':{},'type':'trigger'},out[:min(cutoff,len(out))]))
    return jsonify({'key': "dummy", 'type': 'cluster', 'aux': {'touched': False},'children':out})



@flaskapp.route('/s3/similarcluster',methods=['POST'])
def getSimilarCluster():
    from find_similar import find_similar
    import numpy as np
    exist_triggers = request.json.get('triggers',[])
    threshold = request.json.get('threshold',0.0)
    threshold = float(threshold)
    cutoff = request.json.get('cutoff',10)
    cutoff = int(cutoff) + 1
    limit = request.json.get('limit',c.NUM_DEFAULT_NUM_OF_SENTENCE_IN_TRIGGER)
    limit = int(limit)
    trigger_words = set([trigger.split("_")[-1] for trigger in exist_triggers])
    mtx = np.load(c.JOSHUA_FIND_SIMILAR_MATRIX_NPZ)
    with open(c.JOSHUA_FIND_SIMILAR_MATRIX_INDEX,'r') as fp:
        idx_map = json.load(fp)
    my_db = find_similar.FakeDB(idx_map,mtx)
    out = find_similar.getSimilarClusterInWords(trigger_words, threshold ,cutoff, my_db)
    out = list(map(lambda x:{'trigger':x,'blacklist':[],'trigger_postag':None,'num':c.NUM_DEFAULT_NUM_OF_SENTENCE_IN_TRIGGER},out[:min(cutoff,len(out))]))
    with open(os.path.join(c.FOLDER_SESSION(g.session),'working_corpora.json'),'r') as fp:
        corpora = json.load(fp)
    trigger_key_to_sentence_mapping = dict()
    dao = TriggerAnnotationMongoDBAccessLayer()
    for corpus_name in corpora:
        en = dao.unannotated_sentence_getter(mongo_db,g.session,out,limit,corpus_name,None)
        for i in en:
            key = i['key']
            if key in trigger_key_to_sentence_mapping:
                trigger_key_to_sentence_mapping[key]['aux']['blacklist'].extend(i['aux']['blacklist'])
                trigger_key_to_sentence_mapping[key]['children'].extend(i['children'])
            else:
                trigger_key_to_sentence_mapping[key] = i
    buf = list()
    for _, i in trigger_key_to_sentence_mapping.items():
        buf.append(i)
    ret = list()
    for trigger_entry in buf:
        random.shuffle(trigger_entry['children'])
        trigger_entry['children'] = trigger_entry['children'][:limit]
        ret.append(trigger_entry)
    return jsonify({'key': "dummy", 'type': 'cluster', 'aux': {'touched': False}, 'children': ret})

@flaskapp.route('/s3/grounding',methods=['POST'])
def grounding_to_user_define_event_type():
    from ontology.internal_ontology import return_node_name_joint_path_str,build_internal_ontology_tree_without_exampler
    ontology_tree_root, node_id_to_node_mapping = build_internal_ontology_tree_without_exampler(c.EVENT_YAML_FILE_PATH)
    slash_like_path_to_node_id_mapping = dict()

    for node_id, nodes in node_id_to_node_mapping.items():
        for node in nodes:
            slash_like_path = return_node_name_joint_path_str(node)
            slash_like_path_to_node_id_mapping[slash_like_path] = node_id


    current_event_type = request.json.get('currentEventType')
    grounding_candidate = request.json.get('groundingCandidate')
    if "/" not in grounding_candidate:
        return jsonify({"text":"Grounding is unsuccessful due to missing /","success":False}),200
    grounding_candidate_in_unit = grounding_candidate.split("/")[1:]


    resovled_type = ""
    if "/"+"/".join(grounding_candidate_in_unit) in slash_like_path_to_node_id_mapping:
        resovled_type = "/"+"/".join(grounding_candidate_in_unit)
    elif "/"+"/".join(grounding_candidate_in_unit[:-1]) in slash_like_path_to_node_id_mapping:
        resovled_type = "/"+"/".join(grounding_candidate_in_unit)
    else:
        return jsonify({"text": "Not Groundable due to you're trying to intermediate node does not exists.", "success": False}), 200
    if resovled_type.lower() == "/event":
        return jsonify({"text": "Grounding is unsuccessful due to too general", "success": False}), 200

    # TODO: Actually write to the ontology yaml file

    dao = TriggerAnnotationMongoDBAccessLayer()
    dao.change_event_type(mongo_db,g.session,current_event_type,resovled_type)
    return jsonify({"text": "Successfully ground {} to {}".format(current_event_type,resovled_type), "success": True}), 200
    








@flaskapp.route('/s3/grounding/candidates',methods=['POST'])
def get_grounding_candidates():
    from ontology.internal_ontology import build_internal_ontology_tree,return_node_name_joint_path_str
    ontology_tree_root, node_id_to_node_mapping = build_internal_ontology_tree(c.EVENT_YAML_FILE_PATH,c.DATA_EXAMPLE_JSON_FILE_PATH)
    node_id_to_exemplar_set = dict()
    slash_like_path_to_node_id_mapping = dict()
    node_id_to_slash_like_path_mapping = dict()

    for node_id, nodes in node_id_to_node_mapping.items():
        for node in nodes:
            node_id_to_exemplar_set.setdefault(node_id, set()).update({i.text for i in node.exemplars})
            slash_like_path = return_node_name_joint_path_str(node)
            slash_like_path_to_node_id_mapping[slash_like_path] = node_id
            node_id_to_slash_like_path_mapping[node_id] = slash_like_path
    with open(os.path.join(c.FOLDER_SESSION(g.session), 's3clusters.json'), 'r') as fp:
        session_clusters = json.load(fp)


    for cluster in session_clusters['clusters']:
        if cluster['key'] in slash_like_path_to_node_id_mapping:
            node_id_to_exemplar_set[slash_like_path_to_node_id_mapping[cluster['key']]].update({i['key'] for i in cluster['children']})
    from find_similar import find_similar
    import numpy as np
    mtx = np.load(c.JOSHUA_FIND_SIMILAR_MATRIX_NPZ)
    with open(c.JOSHUA_FIND_SIMILAR_MATRIX_INDEX,'r') as fp:
        idx_map = json.load(fp)
    my_db = find_similar.FakeDB(idx_map,mtx)
    candidate_triggers = request.json.get('triggers', [])
    trigger_words = set([trigger.split("_")[-1] for trigger in candidate_triggers])
    trimmed_pairwise_similarity_mapping = dict()
    for trigger_word in trigger_words:
        list_of_dsts = my_db.find(trigger_word)
        for en in list_of_dsts:
            trimmed_pairwise_similarity_mapping.setdefault(trigger_word,dict()).setdefault(en['dst'],en['score'])
    node_id_to_similarity_calculation = dict()
    for node_id,exemplar_set in node_id_to_exemplar_set.items():
        total = 0
        cnt = 0
        for src_word,dst_words_dict in trimmed_pairwise_similarity_mapping.items():
            for dst_word in dst_words_dict.keys():
                if dst_word in exemplar_set:
                    total += trimmed_pairwise_similarity_mapping[src_word][dst_word]
                    cnt += 1
        node_id_to_similarity_calculation[node_id] = total / max(1,cnt)
    similarity_to_node_id = {v:k for k,v in node_id_to_similarity_calculation.items()}
    candidate_node_id = list()
    for i in sorted(similarity_to_node_id.keys(),reverse=True):
        if node_id_to_slash_like_path_mapping[similarity_to_node_id[i]] == "/Event":
            continue
        candidate_node_id.append(similarity_to_node_id[i])
    return jsonify([node_id_to_slash_like_path_mapping[i] for i in candidate_node_id][:10])




# @flaskapp.route('/s4/metadata',methods=['GET'])
# def PruneUIAnnotationMetaData():
#     with open(os.path.join(c.FOLDER_SESSION(g.session), 'type.json'),'r') as fp:
#         j = json.load(fp)
#     if j['type'] == "argument":
#         with open(os.path.join(c.FOLDER_SESSION(g.session), 'booking.json'),'r') as fp:
#             return jsonify(json.load(fp))
#     elif j['type'] == "event_adjudicated":
#         ret = dict()
#         page_limit = c.EVENT_ADJUDICATED_POOL_PAGINATION_SIZE
#         cnt_dict = dict()
#         for i in mongo_db[c.ANNOTATED_INSTANCE_TABLE_NAME].find({},{'_id':0,'eventType':1}):
#             cnt_dict[i['eventType']] = cnt_dict.get(i['eventType'],0)+1
#         for t,cnt in cnt_dict.items():
#             ret[t] = cnt_dict[t] // page_limit
#         return jsonify(ret)


# @flaskapp.route('/s4/page',methods=['GET','POST'])
# def PruneUIDataStorage():
#     with open(os.path.join(c.FOLDER_SESSION(g.session), 'type.json'),'r') as fp:
#         j = json.load(fp)
#     annotation_type = request.args.get('annotationType', "")
#     page = int(request.args.get('page', -1))
#     if j['type'] == "argument":
#         if request.method=="GET":
#             if len(annotation_type) < 1 or page < 0:
#                 return jsonify({'msg':'Cannot find your annotation file'}),404
#             sentences = list()
#             with open(os.path.join(c.FOLDER_SESSION(g.session),annotation_type,str(page)+".ljson"),'r') as fp:
#                 for i in fp:
#                     sentences.append(json.loads(i))
#             return jsonify({'sentences':sentences})
#         elif request.method=="POST":
#             sentences = request.json.get('sentences')
#             with open(os.path.join(c.FOLDER_SESSION(g.session),annotation_type,str(page)+".ljson"),'w') as fp:
#                 for i in sentences:
#                     fp.write("{}\n".format(json.dumps(i)))
#             return "OK"
#     elif j['type'] == "event_adjudicated":
#         if request.method == "GET":
#             if len(annotation_type) < 1 or page < 0:
#                 return jsonify({'msg':'Cannot find your annotation file'}),404
#             ret = list()
#             for i in mongo_db[c.ANNOTATED_INSTANCE_TABLE_NAME].find({'eventType':annotation_type}).skip(page*c.EVENT_ADJUDICATED_POOL_PAGINATION_SIZE).limit(c.EVENT_ADJUDICATED_POOL_PAGINATION_SIZE):
#                 sentence_info = mongo_db[c.ADJUDICATED_SENTENCE_TABLE_NAME].find_one({'docId':i['docId'],'sentenceId':i['sentenceId']})['sentenceInfo']
#                 ret.append({"key":"{}.{}".format(i['docId'],i['sentenceId']),
#                             "type":"sentence",
#                             "aux":{
#                                 "instanceId":{
#                                     'docId':i['docId'],
#                                     'sentenceId':i['sentenceId'],
#                                     'triggerSentenceTokenizedPosition':i['triggerSentenceTokenizedPosition'],
#                                     'triggerSentenceTokenizedEndPosition':i['triggerSentenceTokenizedEndPosition']
#                                 },
#                                 "sentence":sentence_info['token'],
#                                 "tags":{
#                                     'trigger':[j for j in range(i['triggerSentenceTokenizedPosition'],i['triggerSentenceTokenizedEndPosition']+1)],
#                                     'dummy':[]
#                                 },
#                                 "trigger_lemma":i['trigger'],
#                                 "annotated":True,
#                                 "positive":i['positive'],
#                                 "argument_type":"dummy"
#                             }
#                         })
#             return jsonify({'sentences':ret})
#         elif request.method == "POST":
#             writable = j.get('writable',False)
#             if writable:
#                 ### docId,sentenceId,triggerSentenceTokenizedPosition,triggerSentenceTokenizedEndPosition,annotation_type
#                 sentences = request.json.get('sentences')
#                 for i in sentences:
#                     key = {
#                         'docId': i['aux']['instanceId']['docId'],
#                         'sentenceId': i['aux']['instanceId']['sentenceId'],
#                         'triggerSentenceTokenizedPosition': i['aux']['instanceId']['triggerSentenceTokenizedPosition'],
#                         'triggerSentenceTokenizedEndPosition': i['aux']['instanceId']['triggerSentenceTokenizedEndPosition'],
#                         'eventType': annotation_type
#                     }
#                     potential_entry = mongo_db[c.ANNOTATED_INSTANCE_TABLE_NAME].find_one(key)
#                     potential_entry['positive'] = i['aux']['positive']
#                     mongo_db[c.ANNOTATED_INSTANCE_TABLE_NAME].find_one_and_replace(key, potential_entry)
#             return "OK"

@flaskapp.route('/s4/model_trainable',methods=['GET'])
def query_trainable_status():
    from filelock import FileLock
    lock_path = os.path.join(c.FOLDER_SESSION(g.session),'trigger_argument.lock')
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=1):
            return jsonify(True)
    except:
        return jsonify(False)



@flaskapp.route('/s4/model_training_progress',methods=['GET'])
def model_training_status():
    try:
        with open(os.path.join(c.FOLDER_SESSION(g.session),'nlplingo','progress.json'),'r') as fp:
            return jsonify(json.load(fp))
    except:
        return jsonify({"uptime":0,"progress":"No started","redirect":None,"error_msg":None})


@flaskapp.route('/s4/train_model_from_s3',methods=['POST'])
def init_model_training_from_s3():
    my_python_path = os.path.realpath(os.path.join(sys.executable,os.path.pardir,'python'))
    subprocess.Popen(shlex.split("{} {}/external_executable_wrapper/nlplingo_wrapper.py --mode {} --config_path {} --session_id {}".format(my_python_path,project_root,"train_model_from_s3",c.CONFIG_FILE_PATH,g.session)))
    return jsonify({"uptime":0,"progress":"Scheduled","redirect":"pending_page_for_step_4","error_msg":None})



@flaskapp.route('/s4/train_model_from_s4',methods=['POST'])
def init_model_training_from_s4():
    my_python_path = os.path.realpath(os.path.join(sys.executable, os.path.pardir, 'python'))
    subprocess.Popen(shlex.split("{} {}/external_executable_wrapper/nlplingo_wrapper.py --mode {} --config_path {} --session_id {}".format(my_python_path,project_root,"train_model_from_s4",c.CONFIG_FILE_PATH,g.session)))
    return jsonify({"uptime":0,"progress":"Scheduled","redirect":"pending_page_for_step_4","error_msg":None})

@flaskapp.route('/s4/metadata',methods=['GET'])
def PruneUIAnnotationMetaData():
    dao = ArgumentAnnotationMongoDBAccessLayer()
    return jsonify(dao.get_paginate_argument_metadata(g.session,c.EVENT_ADJUDICATED_POOL_PAGINATION_SIZE,mongo_db))

@flaskapp.route('/s4/page',methods=['GET','POST'])
def HandleAnnotationPaginatedRequest():
    if request.method == "POST":
        sentences = request.json.get('sentences')
        dao = ArgumentAnnotationMongoDBAccessLayer()
        for sentence in sentences:
            sent_obj = TriggerArgumentInstance.fromJSON(sentence)
            event_argument_id = EventArgumentMentionIdentifierTokenIdxBase(sent_obj.instance_identifier_token_idx_base,sent_obj.argument_idx_span)
            dao.modify_annotation(mongo_db,g.session,event_argument_id,sent_obj.event_type,sent_obj.argument_type,sent_obj.touched,sent_obj.marking,True)
        return "OK"
    elif request.method == "GET":
        event_type = request.args.get('eventType')
        argument_type = request.args.get('argumentType')
        page = request.args.get('page', -1)
        try:
            page = int(page)
        except:
            page = -1
        dao = ArgumentAnnotationMongoDBAccessLayer()
        buf = dao.get_paginate_argument_candidates(g.session,event_type,c.EVENT_ADJUDICATED_POOL_PAGINATION_SIZE,page,mongo_db,c.DOCID_TABLE_NAME,argument_type)
        return jsonify(buf)



if __name__ == "__main__":
    flaskapp.run(host='0.0.0.0',port=c.SERVE_PORT)