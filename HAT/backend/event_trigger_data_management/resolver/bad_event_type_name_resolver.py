from event_trigger_data_management.model import EventMention

def resolve_slash(annotated_pool):
    escaped_annotated_pool = dict()
    for root_id in annotated_pool.keys():
        escaped_root_id = root_id.split("/")[-1]

        event_type_to_instance = dict()
        for event_type_id in annotated_pool[root_id].keys():
            escaped_event_type_id = event_type_id.split('/')[-1]
            trigger_set = annotated_pool[root_id][event_type_id]["trigger_set"]
            instance_set = {EventMention(i.docId,
                                         escaped_event_type_id,
                                                  i.sentStartCharOffset,
                                                  i.sentEndCharOffset,
                                                  i.anchorStartCharOffset,
                                                  i.anchorEndCharOffset,
                                                  i.positive,
                                         ) for i in annotated_pool[root_id][event_type_id]["instance_set"]}
            event_type_to_instance[escaped_event_type_id] = {'trigger_set': trigger_set, 'instance_set': instance_set}
        escaped_annotated_pool[escaped_root_id] = event_type_to_instance

    return escaped_annotated_pool