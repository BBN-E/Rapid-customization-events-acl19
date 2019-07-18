import sys,os
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

import json
from utils import serifxml
import re

def my_lemmaizer(string):
    tmp = re.sub(r'[\t\n\r.,_]+','',string)
    tmp = tmp.lower()
    return tmp.replace(" ","_")

def single_document_worker(doc_path):
    serif_doc = serifxml.Document(doc_path)
    for sentence_idx,sentence in enumerate(serif_doc.sentences):
        for sentence_theory in sentence.sentence_theories:
            if len(sentence_theory.token_sequence) < 1:
                continue
            token_start_char_off_to_idx = dict()
            token_end_char_off_to_idx = dict()
            for token_idx,token in enumerate(sentence.token_sequence):
                token_start_char_off_to_idx[token.start_char] = token_idx
                token_end_char_off_to_idx[token.end_char] = token_idx
            for event_mention in sentence.event_mention_set:
                trigger = event_mention.anchor_node.text
                lemmaized_trigger = my_lemmaizer(trigger)
                print(json.dumps({
                    'docId':serif_doc.docid,
                    'sentenceId':{"key":sentence.start_char,"value":sentence.end_char},
                    "trigger":lemmaized_trigger,
                    'triggerPosTag':event_mention.anchor_node.head_tag,
                    "triggerSentenceTokenizedPosition":token_start_char_off_to_idx[event_mention.anchor_node.start_token.start_char],
                    "triggerSentenceTokenizedEndPosition":token_end_char_off_to_idx[event_mention.anchor_node.end_token.end_char]
                }))

def main(serif_folder_path):
    for serif_path in os.listdir(serif_folder_path):
        single_document_worker(os.path.join(serif_folder_path,serif_path))



if __name__ == "__main__":
    serif_folder_path = "/nfs/raid88/u10/users/hqiu/tmp/wm_intervention.generic"
    main(serif_folder_path)