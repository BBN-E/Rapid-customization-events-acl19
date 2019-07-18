import os,sys,json,bisect,shutil
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

from utils import serifxml as serifxml
import collections
import multiprocessing
import re
import subprocess

TokenSpanModel = collections.namedtuple('TokenSpanModel',['tokenIdxInSentence','tokenDocumentStartChar','tokenDocumentEndChar','tokenOriginalText','tokenOriginalPostag'])


def to_tokens(sentence):
    """
    :type sentence: serifxml.Sentence
    :rtype: list[nlplingo.text.text_span.Token]
    """
    ret = []
    """:type: list[nlplingo.text.text_span.Token]"""

    root = sentence.parse.root
    if root is None:
        return ret

    """:type: serifxml.SynNode"""
    for i, t in enumerate(root.terminals):
        t_text = t.text
        t_start = t.start_char
        t_end = t.end_char
        t_pos_tag = t.parent.tag
        ret.append(TokenSpanModel(i,t_start,t_end,t_text,t_pos_tag))
    return ret

def fuzzy_alignment(serif_doc_theory,sentStartOffset,sentEndOffset,triggerStartOffset,triggerEndOffset):
    candidates = dict()
    for idx,sentence in enumerate(serif_doc_theory.sentences):
        nlplingo_token = to_tokens(sentence)
        token_starts = tuple(token.tokenDocumentStartChar for token in nlplingo_token)
        token_ends = tuple(token.tokenDocumentEndChar for token in nlplingo_token)
        if len(token_starts) < 1 or len(token_ends) < 1:
            continue
        if (token_ends[-1] - token_starts[0]) != sentEndOffset - sentStartOffset:
            continue
        else:
            offset = nlplingo_token[0].tokenDocumentStartChar - sentStartOffset
            new_trigger_start_offset = triggerStartOffset + offset
            new_trigger_end_offset = triggerEndOffset + offset
            if new_trigger_start_offset in token_starts and new_trigger_end_offset in token_ends:
                candidates[nlplingo_token[0].tokenDocumentStartChar - sentStartOffset] = sentence
            elif new_trigger_start_offset in token_starts and new_trigger_end_offset - 1 in token_ends:
                candidates[nlplingo_token[0].tokenDocumentStartChar - sentStartOffset] = sentence
    for _,sentence in sorted(candidates.items(),key=lambda k:abs(k[0])):
        return sentence,"Candidate set size is:{}".format(len(candidates.keys()))
    return None,None

def worker_process(docId,docPath,event_argument_list,s3_trigger_queue,err_queue,strict=True):
    try:
        serif = serifxml.Document(docPath)
        sentence_tokensequence_start_to_sentence_idx_mapping = dict()
        sentence_start_to_sentence_idx_mapping = dict()
        for idx,sentence in enumerate(serif.sentences):
            if len(sentence.token_sequence) > 0:
                sentence_tokensequence_start_to_sentence_idx_mapping[sentence.token_sequence[0].start_char] = idx
                sentence_start_to_sentence_idx_mapping[sentence.start_char] = idx
        for _,sentStartOffset,sentEndOffset,triggerStartOffset,triggerEndOffset,argumentStartOffset,argumentEndOffset,argumentType in event_argument_list:
            sentence_start = sentStartOffset
            sentence_end = sentEndOffset
            trigger_start = triggerStartOffset
            trigger_end = triggerEndOffset
            argument_start = argumentStartOffset
            argument_end = argumentEndOffset
            argument_type = argumentType
            if sentence_tokensequence_start_to_sentence_idx_mapping.get(sentence_start,None) is not None:
                sentence = serif.sentences[sentence_tokensequence_start_to_sentence_idx_mapping.get(sentence_start)]
                sentence_start = sentence.start_char
                sentence_end = sentence.end_char
            elif sentence_start_to_sentence_idx_mapping.get(sentence_start,None) is not None:
                sentence = serif.sentences[sentence_start_to_sentence_idx_mapping.get(sentence_start)]
                sentence_start = sentence.start_char
                sentence_end = sentence.end_char
            else:
                if strict is True:
                    err_msg = "Misalignment:{} {} {} {} {} {} {}".format(docId , sentence_start, sentence_end,
                                                                      trigger_start, trigger_end,argument_start,argument_end)
                    # print(err_msg)
                    err_queue.put(err_msg)
                    continue
                fuzzy_sentence, msg = fuzzy_alignment(serif, sentence_start, sentence_end, trigger_start, trigger_end)
                if fuzzy_sentence is None:
                    err_msg = "Misalignment:{} {} {} {} {} {} {}".format(docId, sentence_start, sentence_end,
                                                                      trigger_start, trigger_end,argument_start,argument_end)
                    # print(err_msg)
                    err_queue.put(err_msg)
                    continue
                else:
                    # print(msg)
                    offset = fuzzy_sentence.start_char - sentence_start
                    sentence_start = sentence_start + offset
                    sentence_end = sentence_end + offset
                    trigger_start = trigger_start + offset
                    trigger_end = trigger_end + offset
                    argument_start = argument_start + offset
                    argument_end = argument_end + offset
                    sentence = fuzzy_sentence
            nlplingo_token = to_tokens(sentence)
            token_text = [token.tokenOriginalText for token in nlplingo_token]
            token_starts = [token.tokenDocumentStartChar for token in nlplingo_token]
            token_ends = [token.tokenDocumentEndChar for token in nlplingo_token]
            trigger_start_idx = bisect.bisect(token_starts, trigger_start) - 1
            trigger_end_idx = bisect.bisect(token_ends, trigger_end) - 1
            trigger_idx = list()
            for i in range(trigger_start_idx, trigger_end_idx + 1):
                trigger_idx.append(i)
            argument_idx = list()
            argument_start_idx = bisect.bisect(token_starts, argument_start) - 1
            argument_end_idx = bisect.bisect(token_ends, argument_end) - 1
            for i in range(argument_start_idx,argument_end_idx+1):
                argument_idx.append(i)
            if (len(trigger_idx) < 1):
                err_msg = "Misalignment:{} {} {} {} {} {} {}".format(docId, sentence_start, sentence_end,
                                                                     trigger_start, trigger_end, argument_start,
                                                                     argument_end)
                # print(err_msg)
                err_queue.put(err_msg)
                continue
            if token_starts[trigger_idx[0]] != trigger_start or token_ends[trigger_idx[-1]] != trigger_end:
                if token_ends[trigger_idx[-1]] == trigger_end - 1:
                    trigger_end = trigger_end - 1
                else:
                    err_msg = "Misalignment:{} {} {} {} {} {} {}".format(docId, sentence_start, sentence_end,
                                                                      trigger_start, trigger_end,argument_start,argument_end)
                    # print(err_msg)
                    err_queue.put(err_msg)
                    continue
            trigger_word = " ".join(token_text[trigger_idx[0]:trigger_idx[-1] + 1])
            if trigger_idx[0] - 2 < 0:
                key = token_text[0:5]
            elif trigger_idx[-1] + 2 >= len(token_text):
                key = token_text[-5:]
            else:
                key = token_text[trigger_idx[0] - 2:trigger_idx[-1] + 3]
            key = " ".join(key)
            s3_trigger_queue.put(json.dumps({'key':"{}.{}".format(docId,str((sentence_start,sentence_end))),
                                                                                                     'type':'sentence',
                                                                                                     'aux':{'instanceId':"{} {} {} {} {} {} {} {}".format(docId,sentStartOffset,sentEndOffset,triggerStartOffset,triggerEndOffset,argumentStartOffset,argumentEndOffset,argumentType),
                                                                                                            'sentence':token_text,
                                                                                                            'displayablekey': key,
                                                                                                            'tags':{
                                                                                                                argument_type:argument_idx,
                                                                                                                'trigger':trigger_idx
                                                                                                            },
                                                                                                            'argument_type':argument_type,
                                                                                                            'trigger_lemma':trigger_word,
                                                                                                            'annotated':False,
                                                                                                            'positive':True
                                                                                                            }}))
    except:
        import traceback
        traceback.print_exc()
        err_queue.put(traceback.format_exc())
    finally:
        s3_trigger_queue.close()
        err_queue.close()

def main(input_candidate_file,serifxml_list_file,output_folder):
    docId_to_docPath_mapping = dict()
    shutil.rmtree(output_folder,ignore_errors=True)
    os.makedirs(output_folder,exist_ok=True)
    with open(serifxml_list_file,'r') as fp:
        for i in fp:
            i = i.strip()
            docId = os.path.basename(i)
            docId = docId.replace(".xml","")
            docId_to_docPath_mapping[docId] = i
    manager = multiprocessing.Manager()
    s3_trigger_queue = manager.Queue()
    err_queue = manager.Queue()
    docId_to_eventArgument_mapping = dict()
    with open(input_candidate_file,'r') as fp:
        for idx,i in enumerate(fp):
            i = i.strip()
            docId,sentStart,sentEnd,triggerStart,triggerEnd,argStart,argEnd,argType = i.split(" ")
            docId_to_eventArgument_mapping.setdefault(docId,manager.list()).append(manager.list([docId,int(sentStart),int(sentEnd),int(triggerStart),int(triggerEnd),int(argStart),int(argEnd),argType]))
            if idx % 500 == 0:
                print("Read {} lines of event argument lines.".format(idx))
    with manager.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        print("Begin parallel processing.")
        work_list = list()
        for docId,entry_arr in docId_to_eventArgument_mapping.items():
            work_list.append(pool.apply_async(worker_process, args=(docId,
                                                                    docId_to_docPath_mapping.get(
                                                                        docId),
                                                                    entry_arr,
                                                                    s3_trigger_queue,
                                                                    err_queue, True,)))
        for idx,i in enumerate(work_list):
            if idx % 500 == 0:
                print("We're waiting {} workers to finish".format(len(work_list) - idx))
            i.wait()
    print("We're collecting error reports.")
    while err_queue.empty() is False:
        print(err_queue.get_nowait())
    print("We're writing results.")
    while s3_trigger_queue.empty() is False:
        rec = json.loads(s3_trigger_queue.get_nowait())
        with open(os.path.join(output_folder,rec['aux']['argument_type']+".ljson"),'a') as fp:
            fp.write(json.dumps(rec)+"\n")


def trigger_blacklist_filter(output_folder,word_blacklist):
    for file in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder,file)) and file.endswith(".ljson"):
            output_buf = list()
            with open(os.path.join(output_folder, file), 'r') as fp:
                for i in fp:
                    i = i.strip()
                    j = json.loads(i)
                    trigger_start_idx = j['aux']['tags']['trigger'][0]
                    trigger_end_idx = j['aux']['tags']['trigger'][-1]
                    normalized_trigger_word = " ".join(j['aux']['sentence'][trigger_start_idx:trigger_end_idx+1])
                    normalized_trigger_word = normalized_trigger_word.strip().lower()
                    if normalized_trigger_word in word_blacklist:
                        continue
                    else:
                        output_buf.append(i)
            if len(output_buf) < 1:
                os.remove(os.path.join(output_folder,file))
            else:
                with open(os.path.join(output_folder,file),'w') as fp:
                    for i in output_buf:
                        fp.write("{}\n".format(i))
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def spliter(output_folder,batch_size=200):
    booking = dict()
    for file in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder,file)) and file.endswith(".ljson"):
            current_folder = os.path.join(output_folder,file.replace(".ljson",""))
            current_arg_type_name = file.replace(".ljson","")
            shutil.rmtree(current_folder,ignore_errors=True)
            os.makedirs(current_folder,exist_ok=True)
            with open(os.path.join(output_folder,file),'r') as fp:
                buf = [i.strip() for i in fp]
                for idx,i in enumerate(chunks(buf,batch_size)):
                    booking[current_arg_type_name] = max(booking.get(current_arg_type_name,-1),idx)
                    with open(os.path.join(current_folder,str(idx)+".ljson"),'w') as fp2:
                        for j in i:
                            fp2.write("{}\n".format(j))
    with open(os.path.join(output_folder,'booking.json'),'w') as fp:
        json.dump(booking,fp)
    with open(os.path.join(output_folder,'type.json'),'w') as fp:
        json.dump({'type':'argument'},fp)

def argument_annotation_candidate_generator(config_ins,input_serifxml_list,output_file_path):
    subprocess.check_call([config_ins.BIN_EVENT_ARGUMENT_ANNOTATION_EXAMPLE_GENERATOR,input_serifxml_list,output_file_path])


if __name__ == "__main__":
    # from config import ReadAndConfiguration
    # c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))
    #
    # SCRATCH_PLACE = c.SCRATCH_PLACE
    # input_serifxml_list = c.serifxml_list_path_for_argument
    # big_ljson_buffer = os.path.join(SCRATCH_PLACE,'argument_annotation_big.ljson')
    # argument_annotation_candidate_generator(c,input_serifxml_list,big_ljson_buffer)
    # main(big_ljson_buffer, input_serifxml_list, c.FOLDER_SESSION(c.init_argument_session_id))
    # spliter(c.FOLDER_SESSION(c.init_argument_session_id))
    input_serifxml_list = "/home/hqiu/wm_m12_for_trigger_and_argument_annotation.list"
    big_ljson_buffer = "/home/hqiu/massive/tmp/annotation.span"
    output_folder = "/home/hqiu/massive/tmp/wm_m12_wm_intervention_4_arg"
    main(big_ljson_buffer, input_serifxml_list, output_folder)
    spliter(output_folder)