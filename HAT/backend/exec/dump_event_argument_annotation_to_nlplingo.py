import os,sys,json
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

from event_argument_data_management_old.reader import UIParser
from event_argument_data_management_old.serializer.nlplingo import NLPLINGO
from event_argument_data_management_old.partitioner.num_of_serif_cap_partitioner import NumOfSerifCapPartitioner

argument_role_mapping_table = {
    "Location":"Place",
    "Active":"Actor",
    "Affected":"Actor",
    "FOOD":"Artifact",
    "HYGIENE_TOOL":"Artifact",
    "LIVESTOCK":"Artifact",
    "LIVESTOCK_FEED":"Artifact",
    "MONEY":"Artifact",
    "SEED":"Artifact",
    "WATER":"Artifact",
}


def main(annotation_session_info,output_dir):
    docId_to_docPath_mapping = dict()
    adjudicated_pool_all = set()
    unsupervisored_pool_all = set()
    for i in annotation_session_info:
        serifxml_list_path,UI_session_folder_path = i
        parser = UIParser(serifxml_list_path,UI_session_folder_path)
        docId_to_docPath_mapping.update(parser.unsafe_docId_mapping())
        adjudicated_pool, unsupervisored_pool = parser.parse_to_event_mention_arg_entry(argument_role_mapping_table)
        adjudicated_pool_all.update(adjudicated_pool)
        unsupervisored_pool_all.update(unsupervisored_pool)
    serializer = NLPLINGO(docId_to_docPath_mapping,adjudicated_pool_all,unsupervisored_pool_all,output_dir,NumOfSerifCapPartitioner(2500))
    serializer.serialize()

if __name__ == "__main__":
    # serifxml_list_path = "/home/hqiu/Public/serifxml_list/special/wm_intervention_issue20_96011.list"
    # UI_session_folder_path = "/home/hqiu/ld100/hume/hmi/backend/customized-event/tmp/yeeseng_s4_intervention"
    # output_dir = "/home/hqiu/massive/intervention_wm_ui_to_nlplingo"
    # serifxml_list_path = "/nfs/raid87/u14/users/azamania/runjobs/expts/causeex_pipeline/causeex_m11_0905i/nn_events_serifxml.list"
    # UI_session_folder_path = "/home/hqiu/massive/tmp/event_argument_ui"
    # output_dir = "/home/hqiu/massive/unsupervisor_event_argument_from_causeexm11_to_nlplingo_v5"
    # serifxml_list_path = "/home/hqiu/runjob/expts/causeex_pipeline/causeex_m9_wm_m12/nn_events_serifxml.list"
    # UI_session_folder_path = "/home/hqiu/ld100/hume/hmi/backend/customized-event/tmp/causeex_m9_wm_m12_human_displacement_args"
    # output_dir = "/home/hqiu/Public/event_argument_from_causeexm11_to_nlplingo_p2"
    # output_dir = "/home/hqiu/Public/wm_m12_hackathon_arg_p2"
    # annotation_session_info = [
    #     ("/home/hqiu/Public/serifxml_list/annotation/causeex_m11_1_event_argument.list","/home/hqiu/ld100/hume/hmi/backend/customized-event/tmp/causeex_m11_event_args_p1"),
    #     ("/home/hqiu/Public/serifxml_list/special/wm_intervention_issue20_96011.list","/home/hqiu/ld100/hume/hmi/backend/customized-event/tmp/yeeseng_s4_intervention"),
    #     ("/home/hqiu/Public/serifxml_list/annotation/wm_m12_intervention_argument.list","/home/hqiu/ld100/hume/hmi/backend/customized-event/tmp/wm_m12_intervention_argument_p2"),
    # ]

    from config import ReadAndConfiguration
    c = ReadAndConfiguration(os.path.join("/hat_data", "config_default.json"))

    annotation_session_info = [(c.serifxml_list_path_for_argument,c.FOLDER_SESSION(c.init_argument_session_id))]

    main(annotation_session_info,c.NLPLINGO_ARG_SPAN_OUTPUT_PATH)