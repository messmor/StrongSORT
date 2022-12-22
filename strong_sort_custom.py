"""
@Author: Du Yunhao
@Filename: strong_sort.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 20:14
@Discription: Run StrongSORT
"""
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app_yolov7 import custom_run
from AFLink.AppFreeLink import *
from GSI import GSInterpolation

if __name__ == '__main__':

    ###AFI Link
    model = PostLinker()
    model.load_state_dict(torch.load(opt.path_AFLink))
    dataset = LinkData('', '')
    ###Deepsort
    detection_file = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/yolov7_predictions.npy.gz"
    output_file = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/DeepSORT_Output.txt"
    output_file_link = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/DeepSortLink_Output.txt"
    output_file_link_GSI ="/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/DeepSortLinkGSI_Output.txt"
    min_confidence = 0.2
    nms_max_overlap = 1.0
    min_detection_height = 0
    max_cosine_distance = 0.2
    nn_budget = None
    custom_run(detection_file, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance,
               nn_budget)

    linker = AFLink(
        path_in=output_file,
        path_out=output_file_link,
        model=model,
        dataset=dataset,
        thrT=(-10, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
        thrS=75,
        thrP=0.10  # 0.10 for CenterTrack, FairMOT, TransTrack.
    )
    linker.link()

    GSInterpolation(
        path_in=output_file_link,
        path_out=output_file_link_GSI,
        interval=40,
        tau=10
    )




