# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def gather_sequence_info(sequence_dir, detection_file):
    """
    시퀀스의 정보를 수집하는 함수
    
    Parameters
    ----------
    sequence_dir : str
        MOTChallenge sequence directory 경로
    detection_file : str
        detection file 경로
        
    Returns
    -------
    디렉토리의 시퀀스 정보:

    * sequence_name: 시퀀스 이름
    * image_filenames: 파일 이름
    * detections: MOTChallenge 형식의 검출 파일
    * groundtruth: MOTChallenge 형식의 groud-truth
    * image_size: Image size (height, width).
    * min_frame_idx: 첫번째 인덱스의 이름
    * max_frame_idx: 마지막 인덱스의 이름
    """
    # 이미지 경로
    image_dir = os.path.join(sequence_dir, "img1")
    # 파일이름
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    # 실제값
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    # 검출 파일 불러오기
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    # 실제값 불러오기
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
    # 시퀀스 이미지 읽기
    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None
    # index
    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())
        
    # 시퀀스 정보
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
    """
    주어진 프레임에 대한 detection을 만들기

    Parameters
    ----------
    detection_mat : ndarray
        detection matric은 처음 10개 열은 MOTChallenge 형식이고
        나머지 열은 각 detection과 연관된 벡터가 저장된다.
    frame_idx : int
        프레임 인덱스
    min_height : Optional[int]
        최소 bounding box 높이 / 이것보다 작으면 무시한다.
    Returns
    -------
    List[tracker.Detection]
        detection 반환

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    # detection parsing
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """
    multi-target tracker 실행

    Parameters
    ----------
    sequence_dir : str
        시퀀스 데이터 경로
    detection_file : str
        검출 파일
    output_file : str
        output file 경로 / 추적 결과를 포함
    min_confidence : float
        confidence 임계값 / 이보다 낮은 confidence는 무시하게 된다.
    nms_max_overlap: float
        Maximum detection overlap (NMS 임계값)
    min_detection_height : int
        height 임계값 / 이보다 낮은 height는 무시하게 된다.
    max_cosine_distance : float
        cosine distance metric 임계값
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget is enforced.
    display : bool
        True : 시각화

    """
    # 시퀀스 정보수
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    # cosine distance metric 사용
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    # 프레임 마다 호출
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # detection 생성
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # NMS 실행
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # tracker 업데이트 이 부분에서 detection을 업데이트 해서 객체를 추적한다.
        # 내가 내 코드에 결합하고 싶을 때 사용하는 방법을 이해하기 위해 포스트를 작성했기 떄문에
        # 자세하게 보고싶으면 SORT 포스트를 보자 비슷한거 같다.
        tracker.predict()
        tracker.update(detections)

        # 시각화
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # 결과를 저장한다.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # 객체를 추적하는 시작부분 frame_callback을 매 프레임 호출한다.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # 결과를 저장한다.
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
        default="./tmp/hypotheses.txt")
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


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
