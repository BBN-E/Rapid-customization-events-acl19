import os,shutil,json
from collections import defaultdict

from event_trigger_data_management.serializer import Serializer
from event_trigger_data_management.model.nlplingo_train_trigger_parameter_template import generate_nlplingo_training_parameter_str
from models.nlplingo import Entity_type_str

class NLPLINGOSerializerJOSHUA(Serializer):
    def __init__(self,annotated_pool,docId_to_docPath,output_dir):
        self.annotated_pool = annotated_pool
        self.output_dir = output_dir
        self.docId_to_docPath = docId_to_docPath
    def re_alignment(self):
        raise NotImplementedError
    def serialize(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        for root_id in self.annotated_pool.keys():
            os.makedirs(os.path.join(self.output_dir,root_id))
            current_dir_root = os.path.join(self.output_dir,root_id)
            unlemmatized_trigger_dict = dict()
            for event_type_id in self.annotated_pool[root_id].keys():
                os.makedirs(os.path.join(current_dir_root,event_type_id))
                trigger_set = self.annotated_pool[root_id][event_type_id].get('trigger_set',set())
                instance_set = self.annotated_pool[root_id][event_type_id].get('instance_set',set())
                mentioned_docs = set()
                for event_mention in instance_set:
                    with open(os.path.join(current_dir_root, "{}.sent_spans".format(event_type_id)),
                              'a') as fp:
                        fp.write("{} {} {}\n".format(event_mention.docId, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                    with open(os.path.join(current_dir_root, event_type_id, event_mention.docId) + ".span", 'a') as fp:
                        fp.write("<Event type=\"{}\">\n".format(event_type_id))
                        fp.write("{}\t{}\t{}\n".format(event_type_id, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                        fp.write("anchor\t{}\t{}\n".format(event_mention.anchorStartCharOffset, event_mention.anchorEndCharOffset+1))
                        fp.write("</Event>\n")
                    mentioned_docs.add((event_mention.docId,self.docId_to_docPath.get(event_mention.docId)))
                for docId,docPath in mentioned_docs:
                    with open(os.path.join(current_dir_root, "{}.span_serif_list".format(event_type_id)),
                              'a') as fp:
                        fp.write("SPAN:{} SERIF:{}\n".format(
                            os.path.join(current_dir_root, event_type_id, docId) + ".span",
                            docPath))
                unlemmatized_trigger_dict.setdefault(event_type_id,set()).update(trigger_set)
            with open(os.path.join(current_dir_root, "recovered_keyword.json"), 'w') as fp:
                json.dump([{'event_type': event_type, 'keywords': [w[0] for w in keywords]} for event_type, keywords in
                           unlemmatized_trigger_dict.items()], fp, indent=4)



class NLPLINGOSerializerHAOLING(Serializer):
    def __init__(self,annotated_pool,docId_to_docPath,output_dir):
        self.annotated_pool = annotated_pool
        self.output_dir = output_dir
        self.docId_to_docPath = docId_to_docPath
    def re_alignment(self):
        raise NotImplementedError
    def serialize(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        for root_id in self.annotated_pool.keys():
            os.makedirs(os.path.join(self.output_dir,root_id))
            current_dir_root = os.path.join(self.output_dir,root_id)
            unlemmatized_trigger_dict = dict()
            span_file_buffer = dict()
            mentioned_docs = set()
            event_id_with_event_mention_set = set()
            for event_type_id in self.annotated_pool[root_id].keys():
                trigger_set = self.annotated_pool[root_id][event_type_id].get('trigger_set',set())
                instance_set = self.annotated_pool[root_id][event_type_id].get('instance_set',set())

                if len(instance_set) > 0:
                    event_id_with_event_mention_set.add(event_type_id)

                for event_mention in instance_set:
                    with open(os.path.join(current_dir_root, "{}.sent_spans".format("argument")),
                              'a') as fp:
                        fp.write("{} {} {}\n".format(event_mention.docId, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                    buffer = span_file_buffer.get(event_mention.docId,list())
                    buffer.append("<Event type=\"{}\">\n".format(event_type_id))
                    buffer.append("{}\t{}\t{}\n".format(event_type_id, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                    buffer.append("anchor\t{}\t{}\n".format(event_mention.anchorStartCharOffset, event_mention.anchorEndCharOffset+1))
                    buffer.append("</Event>\n")
                    span_file_buffer[event_mention.docId] = buffer
                    mentioned_docs.add((event_mention.docId,self.docId_to_docPath.get(event_mention.docId)))
                unlemmatized_trigger_dict.setdefault(event_type_id, set()).update(trigger_set)
            for docId,docPath in mentioned_docs:
                with open(os.path.join(current_dir_root, "{}.span_serif_list".format("argument")),
                          'a') as fp:
                    fp.write("SPAN:{} SERIF:{}\n".format(
                        os.path.join(current_dir_root,docId) + ".span",
                        docPath))
            for docId in span_file_buffer.keys():
                with open(os.path.join(current_dir_root,docId) + ".span",'w') as fp:
                    fp.writelines(span_file_buffer[docId])
            with open(os.path.join(current_dir_root, "recovered_keyword.json"), 'w') as fp:
                json.dump([{'event_type': event_type, 'keywords': [w[0] for w in keywords]} for event_type, keywords in
                           unlemmatized_trigger_dict.items()], fp, indent=4)
            with open(os.path.join(current_dir_root, "{}.sent_spans.list".format("argument")),'w') as fp:
                fp.write("{}".format(os.path.join(current_dir_root, "{}.sent_spans".format("argument"))))
            with open(os.path.join(current_dir_root,'domain_ontology.txt'),'w') as fp:
                for event_type in event_id_with_event_mention_set:
                    fp.write("<Event type=\"{}\">\n".format(event_type))
                    fp.write("</Event>\n")
                fp.write(Entity_type_str)

class NLPLINGOSerializerJOSHUA2GEN(Serializer):
    def __init__(self,annotated_pool,docId_to_docPath,output_dir):
        self.annotated_pool = annotated_pool
        self.output_dir = output_dir
        self.docId_to_docPath = docId_to_docPath
    def re_alignment(self):
        raise NotImplementedError
    def serialize(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        unlemmatized_trigger_dict = defaultdict(set)
        consolidated_unlemmatized_trigger_dict = defaultdict(set)
        for root_id in self.annotated_pool.keys():
            os.makedirs(os.path.join(self.output_dir,root_id))
            current_dir_root = os.path.join(self.output_dir,root_id)

            
            span_file_buffer = dict()
            mentioned_docs = set()
            event_id_with_event_mention_set = set()
            for event_type_id in self.annotated_pool[root_id].keys():
                trigger_set = self.annotated_pool[root_id][event_type_id].get('trigger_set',set())
                instance_set = self.annotated_pool[root_id][event_type_id].get('instance_set',set())

                
                child_event_type_id = event_type_id

                event_type_id = root_id


                if len(instance_set) > 0:
                    event_id_with_event_mention_set.add(event_type_id)

                for event_mention in instance_set:
                    with open(os.path.join(self.output_dir, "{}.sent_spans".format(root_id)),
                              'a') as fp:
                        fp.write("{} {} {}\n".format(event_mention.docId, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                    buffer = span_file_buffer.get(event_mention.docId,list())
                    buffer.append("<Event type=\"{}\">\n".format(event_type_id))
                    buffer.append("{}\t{}\t{}\n".format(event_type_id, event_mention.sentStartCharOffset, event_mention.sentEndCharOffset+1))
                    buffer.append("anchor\t{}\t{}\n".format(event_mention.anchorStartCharOffset, event_mention.anchorEndCharOffset+1))
                    buffer.append("</Event>\n")
                    span_file_buffer[event_mention.docId] = buffer
                    mentioned_docs.add((event_mention.docId,self.docId_to_docPath.get(event_mention.docId)))
                unlemmatized_trigger_dict[child_event_type_id].update(trigger_set)
                consolidated_unlemmatized_trigger_dict[event_type_id].update(trigger_set)
            for docId,docPath in mentioned_docs:
                with open(os.path.join(self.output_dir, "{}.span_serif_list".format(root_id)),
                          'a') as fp:
                    fp.write("SPAN:{} SERIF:{}\n".format(
                        os.path.join(self.output_dir,root_id,docId) + ".span",
                        docPath))
          
            for docId in span_file_buffer.keys():
                with open(os.path.join(self.output_dir,root_id
                        ,docId) + ".span",'w') as fp:
                    fp.writelines(span_file_buffer[docId])
        with open(os.path.join(self.output_dir,'domain_ontology.txt'),'w') as fp:
            for event_type_id in self.annotated_pool.keys():
                event = """
<Event type="{}">
<Role>Time</Role>
<Role>Place</Role>
<Role>Active</Role>
<Role>Affected</Role>
<Role>Artifact</Role>
</Event>\n""".format(event_type_id)
                fp.write(event)
        with open(os.path.join(self.output_dir, "recovered_keyword.json"), 'w') as fp:
            json.dump([{'event_type': event_type, 'keywords': [w[0] for w in keywords]} for event_type, keywords in
                       unlemmatized_trigger_dict.items()], fp, indent=4)
        with open(os.path.join(self.output_dir, "consolidated_recovered_keyword.json"), 'w') as fp:
            json.dump([{'event_type': event_type, 'keywords': [w[0] for w in keywords]} for event_type, keywords in
                       consolidated_unlemmatized_trigger_dict.items()], fp, indent=4)

