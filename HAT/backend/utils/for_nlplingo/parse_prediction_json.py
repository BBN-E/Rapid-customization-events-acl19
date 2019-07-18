import re,multiprocessing,os

pattern = re.compile(r"^\((\d+), u?[\'\"]([^\0]+)[\'\"]\)$")

import os,sys,json,shutil
current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir,os.path.pardir))
sys.path.append(project_root)

from utils import serifxml

def docId_to_docPath_mapping(serifxml_list):
    docId_mapping = dict()
    for i in serifxml_list:
        docId = os.path.basename(i)
        filename = docId.split(".")[0]
        docId_mapping[filename] = i
    return docId_mapping

def worker(doc_path,sentence_spans,ret_queue,err_queue):
    try:
        serif_doc = serifxml.Document(doc_path)
        sentence_spans_set = {(i[0],i[1]) for i in sentence_spans}
        ret = list()
        for idx, sentence in enumerate(serif_doc.sentences):
            if len(sentence.token_sequence) > 0:
                if (sentence.token_sequence[0].start_char,sentence.token_sequence[-1].end_char) in sentence_spans_set:
                    ret.append({"docId":serif_doc.docid,
                                "sentenceId":str((sentence.token_sequence[0].start_char,sentence.token_sequence[-1].end_char)),
                                "sentenceInfo":{
                                    "token":[token.text for token in sentence.parse.root.terminals],
                                    "tokenSpan":[{"key":i.start_char,"value":i.end_char} for i in sentence.parse.root.terminals]
                                }
                                })
        ret_queue.put(ret)
    except:
        import traceback
        err_queue.put(traceback.format_exc())

def parse_prediction_json(orig_prediction_json,serifxml_list):
    doc_id_to_sentence_id_request_mapping = dict()
    doc_id_mapping = docId_to_docPath_mapping(serifxml_list)
    for event_type, event in orig_prediction_json.items():
        for _, event_entry in event.items():
            doc_id = event_entry['docId']
            sentence_start_char_off = event_entry['sentenceOffset'][0]
            sentence_end_char_off = event_entry['sentenceOffset'][-1] - 1
            doc_id_to_sentence_id_request_mapping.setdefault(doc_id,set()).add((sentence_start_char_off,sentence_end_char_off))

    manager = multiprocessing.Manager()
    ret_queue = manager.Queue()
    err_queue = manager.Queue()
    with manager.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        print("Begin parallel processing.")
        work_list = list()
        for doc_id,sentence_id_set in doc_id_to_sentence_id_request_mapping.items():
            work_list.append((pool.apply_async(worker,args=(doc_id_mapping.get(doc_id),sentence_id_set,ret_queue,err_queue))))
        for idx,i in enumerate(work_list):
            if idx % 500 == 0:
                print("We're waiting {} workers to finish".format(len(work_list) - idx))
            i.wait()
    print("We're collecting error reports.")
    while err_queue.empty() is False:
        print(err_queue.get_nowait())
    print("We're writing results.")
    doc_id_to_sentence_id_to_sentence_info_mapping = dict()
    while ret_queue.empty() is False:
        sentence_info_set = ret_queue.get()
        for cur in sentence_info_set:
            doc_id_to_sentence_id_to_sentence_info_mapping.setdefault(cur['docId'],dict()).setdefault(cur['sentenceId'],cur['sentenceInfo'])
    ret = list()
    for event_type, event_mention_objs in orig_prediction_json.items():
        for _, event_entry in event_mention_objs.items():
            doc_id = event_entry['docId']
            event_type = event_entry['eventType']
            sentence_start_char_off = event_entry['sentenceOffset'][0]
            sentence_end_char_off = event_entry['sentenceOffset'][-1] - 1
            sentence_id = str((sentence_start_char_off,sentence_end_char_off))
            tokens = doc_id_to_sentence_id_to_sentence_info_mapping[doc_id][sentence_id]["token"]
            token_spans = doc_id_to_sentence_id_to_sentence_info_mapping[doc_id][sentence_id]["tokenSpan"]

            events = [event_entry[i] for i in event_entry if "trigger_" in i]

            for event in events:
                assert len(event['trigger']) == 1
                trigger = event['trigger'][0]
                trigger_start_char_off = token_spans[min(trigger)]['key']
                trigger_end_char_off = token_spans[max(trigger)]['value']
                arguments = {argument_type: argument_arr for argument_type, argument_arr in event.items() if
                            argument_type != "trigger"}
                for argument_type in arguments:
                    for argument in arguments[argument_type]:
                        argument_start_char_off = token_spans[min(argument)]['key']
                        argument_end_char_off = token_spans[max(argument)]['value']
                        new_ui_sentence_obj = {
                            "docId":doc_id,
                            "docPath":doc_id_mapping[doc_id],
                            "sentenceId":sentence_id,
                            "token":tokens,
                            "tokenSpan":token_spans,
                            "eventType":event_type,
                            "triggerSentenceTokenizedPosition":min(trigger),
                            "triggerSentenceTokenizedEndPosition":max(trigger),
                            "argumentSentenceTokenizedPosition":min(argument),
                            "argumentSentenceTokenizedEndPosition":max(argument),
                            "argumentType":argument_type
                        }
                        ret.append(new_ui_sentence_obj)
                new_ui_sentence_obj = {
                            "docId":doc_id,
                            "docPath":doc_id_mapping[doc_id],
                            "sentenceId":sentence_id,
                            "token":tokens,
                            "tokenSpan":token_spans,
                            "eventType":event_type,
                            "triggerSentenceTokenizedPosition":min(trigger),
                            "triggerSentenceTokenizedEndPosition":max(trigger),
                            "argumentSentenceTokenizedPosition": min(trigger),
                            "argumentSentenceTokenizedEndPosition": max(trigger),
                            "argumentType":"trigger"
                        }
                ret.append(new_ui_sentence_obj)
    return ret


if __name__ == "__main__":
    j = "/home/hqiu/runjob/expts/causeex_pipeline/wm_m12.v8.full.v4/nn_events/00002/wm_m12_112018_p4_batch_1/output/prediction.json"
    serifxml_list = "/home/hqiu/runjob/expts/causeex_pipeline/wm_m12.v8.full.v4/nn_events/00002/nn_events_genericity.list"
    import json
    with open(j,'r') as fp:
        j = json.load(fp)

    print(parse_prediction_json(j,serifxml_list))