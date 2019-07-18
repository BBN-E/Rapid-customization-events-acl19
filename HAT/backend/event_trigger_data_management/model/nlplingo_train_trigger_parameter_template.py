
def generate_nlplingo_training_parameter_str(HAOLING_SERIALIZATION_ROOT,event_root_id,nlplingo_out_path):
    return """
cnn.dropout: 0.5
cnn.filter_length: 3
cnn.int_type: int32
cnn.neighbor_dist: 3
cnn.position_embedding_vec_len: 5
cnn.use_bio_index: False
domain: ui
domain_ontology: {0}/{1}/domain_ontology.txt

embedding.embedding_file: /nfs/raid66/u14/users/jfaschin/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.utf8
embedding.missing_token: the
embedding.none_token: .
embedding.none_token_index: 0
embedding.vector_size: 400
embedding.vocab_size: 251236

filelist.train: {0}/{1}/{1}.span_serif_list

home_dir: /nfs/raid87/u15/users/ychan
max_sent_length: 201
negative_trigger_words: /nfs/mercury-04/u40/ychan/event_type_extension/negative_words
#np_span_file: /home/hqiu/Public/np_trigger_test

output_dir: {2}

role.batch_size: 40
role.entity_embedding_vec_length: 10
role.epoch: 20
role.num_feature_maps: 300
role.positive_weight: 10
role.use_head: True
run: run1
trigger.batch_size: 100
trigger.epoch: 10
trigger.num_feature_maps: 200
trigger.positive_weight: 5
trigger_candidate_span_file: {0}/{1}/{1}.sent_spans.list

trigger_model_dir: /nfs/raid87/u13/users/jfaschin/nlplingo-new_types-out

trigger.restrict_none_examples_using_keywords: True
event_keywords: {0}/{1}/recovered_keyword.json
#trigger.spans_to_generate_examples: /nfs/raid87/u13/users/jfaschin/tf_nlplingo_test/yeeseng3_4.anchor_offsets.list

# new parameters
num_batches: 3
experiment_name: cx_11012018
input_directory: 
/nfs/raid87/u15/users/jfaschin/low_shot_keywords/output_11012018/
do_cross_validation: 0
""".format(HAOLING_SERIALIZATION_ROOT,event_root_id,nlplingo_out_path)