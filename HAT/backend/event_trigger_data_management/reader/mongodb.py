import re

from event_trigger_data_management.model import EventMention, SentenceInfoKey
from ontology import internal_ontology


class MongoDBReader():
    def __init__(self,mongo_instance,DOCID_TABLE_NAME,SENTENCE_TABLE_NAME,ANNOTATED_INSTANCE_TABLE_NAME):
        self.mongo_instance = mongo_instance
        self.DOCID_TABLE_NAME = DOCID_TABLE_NAME
        self.SENTENCE_TABLE_NAME = SENTENCE_TABLE_NAME
        self.ANNOTATED_INSTANCE_TABLE_NAME = ANNOTATED_INSTANCE_TABLE_NAME


    def docId_to_docPath_mapping(self,docIdSet=None):
        docId_mapping = dict()
        if docIdSet is not None and len(docIdSet) > 1:
            for docId in docIdSet:
                en = self.mongo_instance[self.DOCID_TABLE_NAME].find_one({'docId':docId})
                docId_mapping[docId] = en['docPath']
        else:
            for i in self.mongo_instance[self.DOCID_TABLE_NAME].find({}):
                docId_mapping[i['docId']] = i['docPath']
        return docId_mapping

    def parse_to_event_mention_entry(self):
        ret = set()
        sentence_info_cache = dict()
        for i in self.mongo_instance[self.ANNOTATED_INSTANCE_TABLE_NAME].find({}):
            sentence_info_key = SentenceInfoKey(i['docId'],i['sentenceId'])
            if sentence_info_cache.get(sentence_info_key,None) is not None:
                en = sentence_info_cache.get(sentence_info_key)
            else:
                en = self.mongo_instance[self.SENTENCE_TABLE_NAME].find_one(dict(sentence_info_key._asdict()))
                assert en is not None
                sentence_info_cache[sentence_info_key] = en
            sentence_start = en['sentenceInfo']['tokenSpan'][0]['key']
            sentence_end = en['sentenceInfo']['tokenSpan'][-1]['value']
            trigger_start = en['sentenceInfo']['tokenSpan'][i['triggerSentenceTokenizedPosition']]['key']
            trigger_end = en['sentenceInfo']['tokenSpan'][i['triggerSentenceTokenizedEndPosition']]['value']
            ret.add(EventMention(i['docId'], i['eventType'], sentence_start, sentence_end, trigger_start,
                                                        trigger_end, i['positive']))
        return ret

    def parsed_and_merged_based_on_carelist(self,ontology_yaml_file,care_list):
        import yaml
        class TreeNode(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.children = list()
                self.parent = kwargs.get('parent', None)

        stack = list()
        node_id_to_node_ptr_mapping = dict()
        with open(ontology_yaml_file, 'r') as fp:
            yaml_file = yaml.load(fp)

        if "Event" in yaml_file[0]:
            root_yaml = yaml_file[0]["Event"]
        else:
            raise ValueError("No Event head found")
        root_node = TreeNode(parent=None,_id="root")
        stack.append(root_node)
        node_id_to_node_ptr_mapping['root'] = root_node

        def dfs_visit(root_yaml, stack,node_ptr_mapping):
            temp_dict = dict()
            for row in root_yaml:
                for k, v in row.items():
                    temp_dict[k] = v
            root_id = temp_dict["_id"]
            current_node = TreeNode(parent=stack[-1],_id=root_id)
            stack[-1].children.append(current_node)
            node_ptr_mapping[root_id] = current_node
            for k, v in temp_dict.items():
                if k.startswith("_"):
                    continue
                else:
                    stack.append(current_node)
                    dfs_visit(v, stack,node_ptr_mapping)
                    stack.pop()
        dfs_visit(root_yaml, stack,node_id_to_node_ptr_mapping)

        care_type_mapping = dict()
        with open(care_list,'r') as fp:
            for i in fp:
                i = i.strip()
                care_type,*blacklist_elems = i.split(",")
                care_type = care_type.strip()
                blacklist_elems = set([i.strip() for i in blacklist_elems])
                care_type_mapping.setdefault(care_type,set()).update(blacklist_elems)

        def flatten(root_ptr):
            ret = set()
            ret.add(root_ptr)
            for child in root_ptr.children:
                ret.update(flatten(child))
            return ret

        annotated_pool = dict()
        docId_to_docPath = dict()
        tuple_stype_re_pattern = re.compile(r"^\((\d+), (\d+)\)$")
        for care_root_str in care_type_mapping.keys():
            blacklist_nodes = set(node_id_to_node_ptr_mapping[i] for i in care_type_mapping.get(care_root_str))
            care_nodes = flatten(node_id_to_node_ptr_mapping[care_root_str])
            care_nodes = care_nodes.difference(blacklist_nodes)

            event_type_to_instance = dict()
            for event_type in care_nodes:
                event_type_id = event_type._id
                trigger_set = set()
                instance_set = set()
                for event_instance in self.mongo_instance[self.ANNOTATED_INSTANCE_TABLE_NAME].find({'eventType':event_type_id}):

                    key = {'docId':event_instance['docId'],
                                          'sentenceId':event_instance['sentenceId'],
                                          'triggerIdxSpan':(event_instance['triggerSentenceTokenizedPosition'],event_instance['triggerSentenceTokenizedEndPosition'])}
                    if docId_to_docPath.get(event_instance['docId'],None) is None:
                        docPath = self.mongo_instance[self.DOCID_TABLE_NAME].find_one({'docId':event_instance['docId']})['docPath']
                        if docPath is None:
                            raise RuntimeError("Fail to look up entry for docId: {}".format(event_instance['docId']))
                        docId_to_docPath[event_instance['docId']] = docPath
                    sentence_info = self.mongo_instance[self.SENTENCE_TABLE_NAME].find_one({'docId':key['docId'],'sentenceId':key['sentenceId']})
                    if sentence_info is None:
                        print("ALERT Sentence Table: docId: {} sentenceId: {} is missing".format(key['docId'],key['sentenceId']))
                        continue
                    sentence_span = (tuple_stype_re_pattern.search(event_instance['sentenceId']).group(1),tuple_stype_re_pattern.search(event_instance['sentenceId']).group(2))
                    trigger_set.add((" ".join(sentence_info['sentenceInfo']['token'][key['triggerIdxSpan'][0]:key['triggerIdxSpan'][-1]+1]),event_instance.get('triggerPosTag',"").replace(".","")))
                    instance_set.add(EventMention(key['docId'],
                                                  event_type_id,
                                                  # int(sentence_span[0]),
                                                  # int(sentence_span[1]),
                                                  sentence_info['sentenceInfo']['tokenSpan'][0]['key'],
                                                  sentence_info['sentenceInfo']['tokenSpan'][-1]['value'],
                                                  int(sentence_info['sentenceInfo']['tokenSpan'][key['triggerIdxSpan'][0]]['key']),
                                                  int(sentence_info['sentenceInfo']['tokenSpan'][key['triggerIdxSpan'][-1]]['value']),
                                                  event_instance['positive']
                                                  ))
                event_type_to_instance.setdefault(event_type_id,dict()).setdefault('trigger_set',set()).update(trigger_set)
                event_type_to_instance.setdefault(event_type_id,dict()).setdefault('instance_set',set()).update(instance_set)
            annotated_pool[care_root_str] = event_type_to_instance
        return annotated_pool,docId_to_docPath

    def parsed_and_merged_based_on_carelist_with_white_list_merging_only(self,ontology_yaml_file,care_list):
        import yaml
        class TreeNode(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.children = list()
                self.parent = kwargs.get('parent', None)

        stack = list()
        node_id_to_node_ptr_mapping = dict()
        with open(ontology_yaml_file, 'r') as fp:
            yaml_file = yaml.load(fp)

        if "Event" in yaml_file[0]:
            root_yaml = yaml_file[0]["Event"]
        else:
            raise ValueError("No Event head found")
        root_node = TreeNode(parent=None,_id="root")
        stack.append(root_node)
        node_id_to_node_ptr_mapping['root'] = root_node

        def dfs_visit(root_yaml, stack,node_ptr_mapping):
            temp_dict = dict()
            for row in root_yaml:
                for k, v in row.items():
                    temp_dict[k] = v
            root_id = temp_dict["_id"]
            current_node = TreeNode(parent=stack[-1],_id=root_id)
            stack[-1].children.append(current_node)
            node_ptr_mapping[root_id] = current_node
            for k, v in temp_dict.items():
                if k.startswith("_"):
                    continue
                else:
                    stack.append(current_node)
                    dfs_visit(v, stack,node_ptr_mapping)
                    stack.pop()
        dfs_visit(root_yaml, stack,node_id_to_node_ptr_mapping)
        def flatten(root_ptr):
            ret = set()
            ret.add(root_ptr)
            for child in root_ptr.children:
                ret.update(flatten(child))
            return ret
        care_type_mapping = dict()
        event_type_mapping_because_of_merging = dict()
        with open(care_list,'r') as fp:
            for i in fp:
                i = i.strip()
                if i.startswith("#"):
                    continue
                model_name,*node_ids = i.split(",")
                model_name = model_name.strip()
                addition_elems = set()
                for j in node_ids:
                    j = j.strip()
                    if j.startswith("*") and j.endswith("*"):
                        true_node_name = j[1:-1]
                        pending_set = flatten(node_id_to_node_ptr_mapping[true_node_name])
                        for k in pending_set:
                            event_type_mapping_because_of_merging[k._id] = true_node_name
                        addition_elems.update(pending_set)
                    else:
                        addition_elems.add(node_id_to_node_ptr_mapping[j])
                        event_type_mapping_because_of_merging[j] = j
                care_type_mapping.setdefault(model_name,set()).update(addition_elems)



        annotated_pool = dict()
        docId_to_docPath = dict()
        tuple_stype_re_pattern = re.compile(r"^\((\d+), (\d+)\)$")
        for model_name in care_type_mapping.keys():
            care_nodes = care_type_mapping[model_name]
            event_type_to_instance = dict()
            for event_type in care_nodes:
                event_type_id = event_type._id
                resolved_event_type_id = event_type_mapping_because_of_merging[event_type_id]
                trigger_set = set()
                instance_set = set()
                for event_instance in self.mongo_instance[self.ANNOTATED_INSTANCE_TABLE_NAME].find({'eventType':event_type_id}):

                    key = {'docId':event_instance['docId'],
                                          'sentenceId':event_instance['sentenceId'],
                                          'triggerIdxSpan':(event_instance['triggerSentenceTokenizedPosition'],event_instance['triggerSentenceTokenizedEndPosition'])}
                    if docId_to_docPath.get(event_instance['docId'],None) is None:
                        docPath = self.mongo_instance[self.DOCID_TABLE_NAME].find_one({'docId':event_instance['docId']})['docPath']
                        if docPath is None:
                            raise RuntimeError("Fail to look up entry for docId: {}".format(event_instance['docId']))
                        docId_to_docPath[event_instance['docId']] = docPath
                    sentence_info = self.mongo_instance[self.SENTENCE_TABLE_NAME].find_one({'docId':key['docId'],'sentenceId':key['sentenceId']})
                    if sentence_info is None:
                        print("ALERT Sentence Table: docId: {} sentenceId: {} is missing".format(key['docId'],key['sentenceId']))
                        continue
                    sentence_span = (tuple_stype_re_pattern.search(event_instance['sentenceId']).group(1),tuple_stype_re_pattern.search(event_instance['sentenceId']).group(2))
                    trigger_set.add((" ".join(sentence_info['sentenceInfo']['token'][key['triggerIdxSpan'][0]:key['triggerIdxSpan'][-1]+1]),event_instance.get('triggerPosTag',"").replace(".","")))
                    instance_set.add(EventMention(key['docId'],
                                                  resolved_event_type_id,
                                                  # int(sentence_span[0]),
                                                  # int(sentence_span[1]),
                                                  sentence_info['sentenceInfo']['tokenSpan'][0]['key'],
                                                  sentence_info['sentenceInfo']['tokenSpan'][-1]['value'],
                                                  int(sentence_info['sentenceInfo']['tokenSpan'][key['triggerIdxSpan'][0]]['key']),
                                                  int(sentence_info['sentenceInfo']['tokenSpan'][key['triggerIdxSpan'][-1]]['value']),
                                                  event_instance['positive']
                                                  ))
                event_type_to_instance.setdefault(resolved_event_type_id,dict()).setdefault('trigger_set',set()).update(trigger_set)
                event_type_to_instance.setdefault(resolved_event_type_id,dict()).setdefault('instance_set',set()).update(instance_set)
            annotated_pool[model_name] = event_type_to_instance
        export_sentence_example_cnt = 0
        for model_name in annotated_pool:
            for resolved_event_type_id in annotated_pool[model_name]:
                export_sentence_example_cnt += len(annotated_pool[model_name][resolved_event_type_id]['instance_set'])
        print(export_sentence_example_cnt)
        return annotated_pool,docId_to_docPath

    def parse_whole_yaml_tree_out_with_auto_resolved_merge(self,ontology_yaml_file):
        ontology_tree_root, node_id_to_node_mapping = internal_ontology.build_internal_ontology_tree_without_exampler(ontology_yaml_file)
        event_type_mapping_because_of_merging = dict()
        def flatten(root_ptr):
            ret = set()
            ret.add(root_ptr)
            for child in root_ptr.children:
                ret.update(flatten(child))
            return ret

        for node_id, nodes in node_id_to_node_mapping.items():
            for node in nodes:
                possible_nn_node = internal_ontology.find_lowest_nn_node(node)
                if possible_nn_node is None:
                    print("WARNING: Unresolveable type {}".format(node_id))
                else:
                    event_type_mapping_because_of_merging.setdefault(node_id, set()).add(possible_nn_node)
        annotated_pool = dict()
        docId_to_docPath = dict()
        tuple_stype_re_pattern = re.compile(r"^\((\d+), (\d+)\)$")

        model_name = "DummyRoot"
        care_nodes = event_type_mapping_because_of_merging.keys()
        event_type_to_instance = dict()
        for event_type in care_nodes:
            for event_type_id in {i.original_key for i in node_id_to_node_mapping[event_type]}:
                for resolved_event_type_id in {i.original_key for i in
                                               event_type_mapping_because_of_merging[event_type_id]}:
                    trigger_set = set()
                    instance_set = set()
                    for event_instance in self.mongo_instance[self.ANNOTATED_INSTANCE_TABLE_NAME].find(
                            {'eventType': event_type_id}):

                        key = {'docId': event_instance['docId'],
                               'sentenceId': event_instance['sentenceId'],
                               'triggerIdxSpan': (event_instance['triggerSentenceTokenizedPosition'],
                                                  event_instance['triggerSentenceTokenizedEndPosition'])}
                        if docId_to_docPath.get(event_instance['docId'], None) is None:
                            docPath = \
                                self.mongo_instance[self.DOCID_TABLE_NAME].find_one({'docId': event_instance['docId']})[
                                    'docPath']
                            if docPath is None:
                                raise RuntimeError(
                                    "Fail to look up entry for docId: {}".format(event_instance['docId']))
                            docId_to_docPath[event_instance['docId']] = docPath
                        sentence_info = self.mongo_instance[self.SENTENCE_TABLE_NAME].find_one(
                            {'docId': key['docId'], 'sentenceId': key['sentenceId']})
                        if sentence_info is None:
                            print("ALERT Sentence Table: docId: {} sentenceId: {} is missing".format(key['docId'],
                                                                                                     key['sentenceId']))
                            continue
                        sentence_span = (tuple_stype_re_pattern.search(event_instance['sentenceId']).group(1),
                                         tuple_stype_re_pattern.search(event_instance['sentenceId']).group(2))
                        trigger_set.add((" ".join(sentence_info['sentenceInfo']['token'][
                                                  key['triggerIdxSpan'][0]:key['triggerIdxSpan'][-1] + 1]),
                                         event_instance.get('triggerPosTag', "").replace(".", "")))
                        instance_set.add(EventMention(key['docId'],
                                                      resolved_event_type_id,
                                                      # int(sentence_span[0]),
                                                      # int(sentence_span[1]),
                                                      sentence_info['sentenceInfo']['tokenSpan'][0]['key'],
                                                      sentence_info['sentenceInfo']['tokenSpan'][-1]['value'],
                                                      int(sentence_info['sentenceInfo']['tokenSpan'][
                                                              key['triggerIdxSpan'][0]]['key']),
                                                      int(sentence_info['sentenceInfo']['tokenSpan'][
                                                              key['triggerIdxSpan'][-1]]['value']),
                                                      event_instance['positive']
                                                      ))
                    event_type_to_instance.setdefault(event_type_id, dict()).setdefault('trigger_set', set()).update(
                        trigger_set)
                    event_type_to_instance.setdefault(resolved_event_type_id, dict()).setdefault('instance_set',
                                                                                                 set()).update(
                        instance_set)
        annotated_pool[model_name] = event_type_to_instance
        export_sentence_example_cnt = 0
        for model_name in annotated_pool:
            for resolved_event_type_id in annotated_pool[model_name]:
                export_sentence_example_cnt += len(annotated_pool[model_name][resolved_event_type_id].get('instance_set',set()))
        print(export_sentence_example_cnt)
        return annotated_pool,docId_to_docPath


    def parse_db_session_table_directly_out(self,session):
        annotated_pool = dict()
        annotated_pool["DummyRoot"] = dict()
        current_root = annotated_pool["DummyRoot"]
        docId_to_docObj = dict()
        for i in self.mongo_instance['{}_trigger_marking'.format(session)].find({}):
            annotation_type = i['eventType']
            if docId_to_docObj.get(i['docId'], None) is None:
                en = self.mongo_instance[self.DOCID_TABLE_NAME].find_one({'docId': i['docId']})
                docId_to_docObj[i['docId']] = en
            sentence_info = self.mongo_instance['{}_sentence_info'.format(docId_to_docObj[i['docId']]['corpusName'])].find_one(
                {'docId': i['docId'], 'sentenceId': i['sentenceId']})
            current_root.setdefault(annotation_type, dict()).setdefault('trigger_set', set()).add(tuple((" ".join(
                sentence_info['sentenceInfo']['token'][
                i['triggerSentenceTokenizedPosition']:i['triggerSentenceTokenizedEndPosition'] + 1]), None)))
            current_root.setdefault(annotation_type, dict()).setdefault('instance_set', set()).add(EventMention(
                i['docId'],
                annotation_type,
                sentence_info['sentenceInfo']['tokenSpan'][0]['key'],
                sentence_info['sentenceInfo']['tokenSpan'][-1]['value'],
                sentence_info['sentenceInfo']['tokenSpan'][i['triggerSentenceTokenizedPosition']]['key'],
                sentence_info['sentenceInfo']['tokenSpan'][i['triggerSentenceTokenizedEndPosition']]['value'],
                True
            ))
        return annotated_pool
