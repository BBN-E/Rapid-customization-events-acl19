import os,sys,json,shutil,collections
current_script_path = __file__
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir,os.path.pardir))
sys.path.append(project_root)

from nlp.EventMention import EventMentionInstanceIdentifierCharOffsetBase
from models.nlplingo import generate_nlplingo_training_parameter_str,Entity_type_str
from nlp.common import Marking


def nlplingo_serializer_flatten_flavor(parsed_dict_from_mongo, output_folder, output_negative_examples = True):
    shutil.rmtree(output_folder,ignore_errors=True)
    os.makedirs(output_folder)
    doc_id_to_doc_path = dict()
    doc_id_to_event_mention_to_event_arg = dict()
    sent_span_set = set()
    event_mention_id_to_trigger_word = dict()
    for k,argument_set in parsed_dict_from_mongo.items():
        doc_path, event_mention_id, event_type,trigger_word = k
        assert isinstance(event_mention_id,EventMentionInstanceIdentifierCharOffsetBase)
        doc_id_to_doc_path[event_mention_id.doc_id] = doc_path
        for argument in argument_set:
            doc_id_to_event_mention_to_event_arg.setdefault(event_mention_id.doc_id,dict()).setdefault((event_mention_id,event_type),set()).add(argument)
        event_mention_id_to_trigger_word[event_mention_id] = trigger_word
    current_working_folder = output_folder
    event_type_to_trigger_word_mapping = dict()
    span_file_serif_path_output_buffer = dict()
    argument_type_set = set()
    for doc_id in doc_id_to_event_mention_to_event_arg.keys():
        potential_span_file_path = os.path.join(current_working_folder, doc_id + ".span")
        serifxml_path = doc_id_to_doc_path.get(doc_id)
        for event_mention_id,event_mention_type in doc_id_to_event_mention_to_event_arg[doc_id].keys():
            assert isinstance(event_mention_id,EventMentionInstanceIdentifierCharOffsetBase)
            output_buffer = list()

            should_output = False
            trigger_positive = True
            for event_arg in doc_id_to_event_mention_to_event_arg[doc_id][(event_mention_id,event_mention_type)]:
                should_output = True
                argument_type, argument_start_char, argument_end_char, touched,positive = event_arg
                if argument_type != "trigger" and positive == Marking.POSITIVE:
                    argument_type_set.add(argument_type)
                    output_buffer.append("{}/{} {} {}\n".format(event_mention_type,argument_type,argument_start_char,argument_end_char+1))
                if argument_type == "trigger" and positive == Marking.NEGATIVE:
                    trigger_positive = False

            if output_negative_examples is True:
                ptr_to_span_file_output_buffer = span_file_serif_path_output_buffer.setdefault(
                    (potential_span_file_path, serifxml_path, doc_id), list())
                sent_span_set.add((doc_id, event_mention_id.sentence_char_off_span.start_offset,
                                   event_mention_id.sentence_char_off_span.end_offset + 1))

            if should_output is True and trigger_positive is True:
                if output_negative_examples is False:
                    ptr_to_span_file_output_buffer = span_file_serif_path_output_buffer.setdefault(
                        (potential_span_file_path, serifxml_path, doc_id), list())
                    sent_span_set.add((doc_id, event_mention_id.sentence_char_off_span.start_offset,
                                       event_mention_id.sentence_char_off_span.end_offset + 1))

                ptr_to_span_file_output_buffer.append("<Event type=\"{}\">\n".format(event_mention_type))
                ptr_to_span_file_output_buffer.append("{} {} {}\n".format(event_mention_type,event_mention_id.sentence_char_off_span.start_offset,event_mention_id.sentence_char_off_span.end_offset+1))

                if len(output_buffer) > 0:
                    ptr_to_span_file_output_buffer.extend(output_buffer)
                ptr_to_span_file_output_buffer.append("anchor {} {}\n".format(event_mention_id.trigger_char_off_span.start_offset,event_mention_id.trigger_char_off_span.end_offset+1))
                ptr_to_span_file_output_buffer.append("</Event>\n")
                event_type_to_trigger_word_mapping.setdefault(event_mention_type,set()).add(event_mention_id_to_trigger_word[event_mention_id])

    # Do Some check on project before you change this name "argument.span_serif_list", "argument.sent_spans","argument.sent_spans.list","argument.sent_spans"

    with open(os.path.join(current_working_folder, 'argument.span_serif_list'),'w') as booking_fp:
        for k,v in span_file_serif_path_output_buffer.items():
            span_file_path,serifxml_path,doc_id = k
            booking_fp.write("SPAN:{} SERIF:{}\n".format(os.path.join(current_working_folder, doc_id + ".span"),
                                                 doc_id_to_doc_path.get(doc_id)))
            with open(span_file_path,'w') as span_fp:
                for output_item in v:
                    span_fp.write(output_item)




    with open(os.path.join(current_working_folder, 'argument.sent_spans'), 'w') as fp:
        for i in sent_span_set:
            fp.write("{} {} {}\n".format(i[0], i[1], i[2]))
    with open(os.path.join(current_working_folder,'argument.sent_spans.list'),'w') as fp:
        fp.write(os.path.join(current_working_folder, 'argument.sent_spans'))
    with open(os.path.join(current_working_folder, 'domain_ontology.txt'), 'w') as fp:
        for event_type in event_type_to_trigger_word_mapping.keys():
            fp.write("<Event type=\"{}\">\n".format(event_type))
            for argument_type in argument_type_set:
                fp.write("<Role>{}</Role>\n".format(argument_type))
            # fp.write("<Role>Time</Role>\n")
            # fp.write("<Role>Place</Role>\n")
            # fp.write("<Role>Active</Role>\n")
            # fp.write("<Role>Affected</Role>\n")
            # fp.write("<Role>Artifact</Role>\n")
            fp.write("</Event>\n")
        fp.write(Entity_type_str)
    os.makedirs(os.path.join(current_working_folder, 'nlplingo_out'), exist_ok=True)
    with open(os.path.join(current_working_folder, 'params'), 'w') as fp:
        fp.write(
            generate_nlplingo_training_parameter_str(current_working_folder,os.path.join(current_working_folder,'nlplingo_out')))
    shutil.copy(os.path.join(current_working_folder, 'domain_ontology.txt'),os.path.join(current_working_folder,'nlplingo_out'))
    with open(os.path.join(current_working_folder,'recovered_keyword.json'),'w') as fp:
        json.dump([{'event_type': event_type, 'keywords': [w for w in keywords]} for event_type, keywords in
                   event_type_to_trigger_word_mapping.items()], fp, indent=4)

if __name__ == "__main__":
    import pymongo
    from data_access_layer.argument_annotation.mongodb import ArgumentAnnotationMongoDBAccessLayer
    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join(project_root, "config_default.json"))
    mongo_instance = pymongo.MongoClient(c.DB_MONGOURI).get_database(c.DB_NAME)
    dao = ArgumentAnnotationMongoDBAccessLayer()
    session = "hqiu_argument_test"
    doc_id_table_name = c.DOCID_TABLE_NAME
    annotations = dao.dump_annotation_out(mongo_instance,session,doc_id_table_name)
    nlplingo_serializer_flatten_flavor(annotations, "/home/hqiu/massive/tmp/serialization_test")