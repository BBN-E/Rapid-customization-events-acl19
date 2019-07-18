

def insert_or_update_sentence(mongo_instance,sentence_table_name,doc_id,sentence_id,sentence_info):
    en = mongo_instance[sentence_table_name].find_one({'docId':doc_id,'sentenceId':sentence_id})
    if en is None:
        mongo_instance[sentence_table_name].insert_one({'docId':doc_id,'sentenceId':sentence_id,'sentenceInfo':sentence_info,'fullSentenceText':" ".join(sentence_info['token'])})
    else:
        pass
        # mongo_instance[sentence_table_name].find_one_and_replace({'docId':doc_id,'sentenceId':sentence_id},{'docId':doc_id,'sentenceId':sentence_id,'sentenceInfo':sentence_info,'fullSentenceText':" ".join(sentence_info['token'])})

