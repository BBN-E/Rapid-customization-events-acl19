import os,sys,json
# def generate_nlplingo_training_parameter_str(SERIALIZATION_ROOT,nlplingo_out_path):
#     return """
# cnn.dropout: 0.5
# cnn.filter_length: 3
# cnn.int_type: int32
# cnn.neighbor_dist: 3
# cnn.position_embedding_vec_len: 5
# cnn.use_bio_index: False
# domain: ui
# domain_ontology: {0}/domain_ontology.txt
#
# embedding.embedding_file: /nfs/raid66/u14/users/jfaschin/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.utf8
# embedding.missing_token: the
# embedding.none_token: .
# embedding.none_token_index: 0
# embedding.vector_size: 400
# embedding.vocab_size: 251236
#
# filelist.train: {0}/argument.span_serif_list
# filelist.dev: {0}/argument.span_serif_list
#
# filelist_dir: /nfs/raid87/u15/users/ychan/wm/experiments/ui_032118
# home_dir: /nfs/raid87/u15/users/ychan
# max_sent_length: 201
# negative_trigger_words: /nfs/mercury-04/u40/ychan/event_type_extension/negative_words
# #np_span_file: /home/hqiu/Public/np_trigger_test
# event_keywords: {0}/recovered_keyword.json
# output_dir: {1}
#
#
# role.batch_size: 40
# role.entity_embedding_vec_length: 10
# role.epoch: 20
# role.num_feature_maps: 300
# role.positive_weight: 10
# role.use_head: True
# role.use_event_embedding: False
# run: run1
# trigger.batch_size: 100
# trigger.epoch: 10
# trigger.num_feature_maps: 200
# trigger.positive_weight: 5
# trigger_candidate_span_file: {0}/argument.sent_spans.list
#
# trigger_model_dir: /nfs/raid87/u13/users/jfaschin/nlplingo-new_types-out
#
# trigger.restrict_none_examples_using_keywords: True
#
# trigger.spans_to_generate_examples: /nfs/raid87/u13/users/jfaschin/tf_nlplingo_test/yeeseng3_4.anchor_offsets.list
#
# # new parameters
# num_batches: 3
# experiment_name: cx_11012018
# input_directory:
# /nfs/raid87/u15/users/jfaschin/low_shot_keywords/output_11012018/
# do_cross_validation: 0
# """.format(SERIALIZATION_ROOT,nlplingo_out_path)


def generate_nlplingo_training_parameter_str(SERIALIZATION_ROOT,nlplingo_out_path):

    json_param = {
        "trigger.restrict_none_examples_using_keywords": False,
        "data": {
            "train": {"filelist": os.path.join(SERIALIZATION_ROOT, 'argument.span_serif_list')},
            "dev": {"filelist": os.path.join(SERIALIZATION_ROOT, 'argument.span_serif_list')}
        },
        "embeddings": {
            "embedding_file": "/nfs/raid66/u14/users/jfaschin/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.utf8",
            "missing_token": "the",
            "none_token": ".",
            "vector_size": 400,
            "vocab_size": 251236,
            "none_token_index": 0
        },
        "extractors": [
            {
                "domain_ontology": os.path.join(nlplingo_out_path,'domain_ontology.txt'),
                "hyper-parameters": {
                    "batch_size": 100,
                    "cnn_filter_lengths": [
                        3
                    ],
                    "dropout": 0.5,
                    "entity_embedding_vector_length": 5,
                    "epoch": 10,
                    "fine-tune_epoch": 0,
                    "neighbor_distance": 3,
                    "number_of_feature_maps": 300,
                    "position_embedding_vector_length": 10,
                    "positive_weight": 10

                },
                "max_sentence_length": 201,
                "model_file": os.path.join(nlplingo_out_path,'argument.hdf'),
                "model_flags": {
                    "use_trigger": True,
                    "use_head": True,
                    "use_event_embedding":False,
                    "train_embeddings":False
                },
                "int_type": "int32",
                "model_type": "event-argument_cnn"
            }
        ],
        "negative_trigger_words": "/nfs/raid84/u12/ychan/u40/event_type_extension/negative_words",
        "train.score_file": os.path.join(nlplingo_out_path,'train.score'),
        "test.score_file": os.path.join(nlplingo_out_path,'test.score'),
        "save_model": True
    }

    return json.dumps(json_param,sort_keys=True,indent=4)

Entity_type_str = """
<Entity subtype="Contact-Info">
<Entity subtype="Crime">
<Entity subtype="FAC.Airport">
<Entity subtype="FAC.Building-Grounds">
<Entity subtype="FAC.Path">
<Entity subtype="FAC.Plant">
<Entity subtype="FAC.Subarea-Facility">
<Entity subtype="GPE.Continent">
<Entity subtype="GPE.County-or-District">
<Entity subtype="GPE.GPE-Cluster">
<Entity subtype="GPE.Nation">
<Entity subtype="GPE.Population-Center">
<Entity subtype="GPE.Special">
<Entity subtype="GPE.State-or-Province">
<Entity subtype="Job-Title">
<Entity subtype="LOC.Address">
<Entity subtype="LOC.Boundary">
<Entity subtype="LOC.Celestial">
<Entity subtype="LOC.Land-Region-Natural">
<Entity subtype="LOC.Region-General">
<Entity subtype="LOC.Region-International">
<Entity subtype="LOC.Water-Body">
<Entity subtype="Numeric">
<Entity subtype="ORG.Commercial">
<Entity subtype="ORG.Educational">
<Entity subtype="ORG.Entertainment">
<Entity subtype="ORG.Government">
<Entity subtype="ORG.Media">
<Entity subtype="ORG.Medical-Science">
<Entity subtype="ORG.Non-Governmental">
<Entity subtype="ORG.Religious">
<Entity subtype="ORG.Sports">
<Entity subtype="PER.Group">
<Entity subtype="PER.Indeterminate">
<Entity subtype="PER.Individual">
<Entity subtype="Sentence">
<Entity subtype="Time">
<Entity subtype="VEH.Air">
<Entity subtype="VEH.Land">
<Entity subtype="VEH.Subarea-Vehicle">
<Entity subtype="VEH.Underspecified">
<Entity subtype="VEH.Water">
<Entity subtype="WEA.Biological">
<Entity subtype="WEA.Blunt">
<Entity subtype="WEA.Chemical">
<Entity subtype="WEA.Exploding">
<Entity subtype="WEA.Nuclear">
<Entity subtype="WEA.Projectile">
<Entity subtype="WEA.Sharp">
<Entity subtype="WEA.Shooting">
<Entity subtype="WEA.Underspecified">
<Entity subtype="ART.Oil">
<Entity subtype="ART.NaturalGas">


<Entity type="Contact-Info">
<Entity type="Crime">
<Entity type="FAC">
<Entity type="GPE">
<Entity type="Job-Title">
<Entity type="LOC">
<Entity type="OTH">
<Entity type="UNDET">
<Entity type="Numeric">
<Entity type="ORG">
<Entity type="PER">
<Entity type="Sentence">
<Entity type="Time">
<Entity type="VEH">
<Entity type="WEA">
<Entity type="HUMAN_RIGHT">
<Entity type="SEXUAL_VIOLENCE">
<Entity type="HYGIENE_TOOL">
<Entity type="FARMING_TOOL">
<Entity type="DELIVERY_KIT">
<Entity type="INSECT_CONTROL">
<Entity type="LIVESTOCK_FEED">
<Entity type="VETERINARY_SERVICE">
<Entity type="FISHING_TOOL">
<Entity type="STATIONARY">
<Entity type="TIMEX2">
<Entity type="ART">
<Entity type="FOOD">
<Entity type="LIVESTOCK">
<Entity type="MONEY">
<Entity type="SEED">
<Entity type="WATER">
<Entity type="CROP">
<Entity type="REFUGEE">
<Entity type="MEDICAL">
<Entity type="FERTILIZER">
<Entity type="THERAPEUTIC_FEEDING">
"""


def generate_nlplingo_decoding_parameter_str(model_folder,file_list,nlplingo_out_path):
    return """
# A canonical params file for ACE pair model

#### expt config
run: run1
trigger_model_dir: {0}/

#### hyper-params
trigger.positive_weight: 5
trigger.epoch: 10
trigger.batch_size: 100

role.positive_weight: 10
role.epoch: 20
role.batch_size: 40

#### hyper-params that are currently fixed
cnn.neighbor_dist: 3
cnn.use_bio_index: False
cnn.int_type: int32
cnn.position_embedding_vec_len: 5
cnn.filter_length: 3
cnn.dropout: 0.5
trigger.num_feature_maps: 200
role.num_feature_maps: 300
role.use_event_embedding: False
role.use_head: True
role.entity_embedding_vec_length: 10

#### following are fixed
#### NOTE!! we use 'filelist.test' just to make sure this sample runjobs sequence runs fast.
#### You should change it to using: train.filelist, dev.filelist, test.filelist
filelist.test: {1}

domain: ui
domain_ontology: {0}/domain_ontology.txt

embedding.embedding_file: /nfs/raid87/u14/CauseEx/nn_event_models/shared/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.spaceSep.utf8
embedding.vector_size: 400
embedding.vocab_size: 251236
embedding.none_token: .
embedding.missing_token: the
embedding.none_token_index: 0

max_sent_length: 201

negative_trigger_words: /nfs/mercury-04/u40/ychan/event_type_extension/negative_words
# following will be over-written by runjobs. nlplingo will write training model and score files to here, and also read model file from here during decoding
output_dir: {2}

""".format(model_folder,file_list,nlplingo_out_path)