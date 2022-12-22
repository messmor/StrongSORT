# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import gzip
import os
import random
import cv2
import numpy as np
from pathlib import Path
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from opts import opt
from time import time

def convert_YOLO_2_MOT(yolo_data):
    '''converts yolo data which is an array of lists of detections of the form '''
    '''[[x_max, x_min, y_max, y_min, conf, cls], ...]'''
    '''into a numpy array of the with an entry for each detection and the MOT data format'''
    ''' <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> '''
    assert isinstance(yolo_data, np.ndarray) or isinstance(yolo_data, list)

    MOT_Data = []

    total_frames = len(yolo_data)
    for f_num in range(total_frames):
        num_det = len(yolo_data[f_num])
        for d in range(num_det):
            det = yolo_data[f_num][d]
            if det:
                ##yolo data
                x_max = det[0]
                x_min = det[1]
                y_max = det[2]
                y_min = det[3]
                conf = det[4]
                cls = det[5]
                ##MOT data
                id = -1
                bb_left = x_min
                bb_top = y_min
                bb_width = abs(x_max-x_min)
                bb_height = abs(y_max-y_min)
                x,y,z = -1, -1 , -1
                if bb_width!=0 and bb_height!=0:
                    MOT_Data.append([f_num, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z])



    MOT_Data = np.asarray(MOT_Data)

    return MOT_Data







def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def custom_run(detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget):
    st = time()
    #load yolov7 format detections
    detection_file = Path(detection_file)
    if not detection_file.is_file():
        ValueError(f"detection file: {str(detection_file)}!")
    if detection_file.suffix == ".gz":
        file = gzip.GzipFile(detection_file.as_posix(),"r")
        detections = np.load(file, allow_pickle=True)
    else:
        detections = np.load(detection_file.as_posix(), allow_pickle=True)


    #load used seq info
    num_frames = len(detections)
    detections = convert_YOLO_2_MOT(detections)
    print("MOT detections shape", detections.shape)

    seq_info = dict(detections=detections, min_frame_idx=1, max_frame_idx=num_frames)
    et = time()

    print(f"runtime up to data conversion is {et-st} seconds!")


    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric)
    results = []

    def frame_callback(frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)

        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)

        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        # Update tracker.
        if opt.ECC:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)

        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def create_output_video(source_video_path, sort_data_path):
    if not Path(sort_data_path).is_file():
        ValueError(f" sort data {sort_data_path} does not exist!")

    if not Path(source_video_path).is_file():
        ValueError(f" source video path {source_video_path} does not exist!")

    sort_data = np.loadtxt(sort_data_path, delimiter=',')
    max_frame = int(np.amax(sort_data[:,0]))
    det_list = []

    for f_num in range(1,max_frame+1):
        f_det = sort_data[np.where(sort_data[:,0] == f_num)]
        det_list.append(f_det)


    ####create movie
    vid = cv2.VideoCapture(source_video_path)
    ret, frame = vid.read()
    count = 0
    colors = [[255,0,0],[0,0,255],[0,255,0]]
    while ret and count <= max_frame:
        dets = det_list[count]
        if len(dets) > 0:
            for d_i, det in enumerate(dets):
                bb_left, bb_top, bb_width, bb_height = det[2:6]
                box = [int(bb_width+bb_left), int(bb_left),int(bb_height+bb_top), int(bb_top)]
                plot_one_box(x=box,img=frame, color=colors[d_i % 3],label=f"id {det[1]}")

        cv2.imshow('', frame)
        cv2.waitKey(50)


        ret, frame = vid.read()
        count+=1

    vid.release()













if __name__ == "__main__":
    # args = parse_args()

    # run(
    #     args.sequence_dir, args.detection_file, args.output_file,
    #     args.min_confidence, args.nms_max_overlap, args.min_detection_height,
    #     args.max_cosine_distance, args.nn_budget, args.display)

    # detection_file = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/yolov7_predictions.npy.gz"
    # output_file = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/StrongSORT_Output.txt"
    # min_confidence = 0.5
    # nms_max_overlap = 2.0
    # min_detection_height = 0
    # max_cosine_distance = 0.2
    # nn_budget = None
    #
    # custom_run(detection_file,output_file,min_confidence,nms_max_overlap,min_detection_height, max_cosine_distance, nn_budget)


    source_video_path = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/undistorted_Alec Jogging_DV1.mov"
    sort_data_path = "/media/mitchell/ssd2/Mocap_Data/MoCap Data/Alec/Capture 1/Alec Jogging/DeepSortLinkGSI_Output.txt"
    create_output_video(source_video_path,sort_data_path)
