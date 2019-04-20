# Deep SORT

## Introduction
- [SORT](https://github.com/abewley/sort) 확장판
- SORT 논문 : [arXiv preprint](https://arxiv.org/abs/1703.07402)

## Dependencies
```
NumPy
sklearn
OpenCV
```
Additionally, feature generation requires TensorFlow (>= 1.0).

## Installation

저장소 clone:
```
git clone https://github.com/nwojke/deep_sort.git
```

미리 생성된 검출과 CNN checkpoint 파일 다운로드 [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

*NOTE:* 미리 생성된 검출은 아래 논문에서 발췌:
```
F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, J. Yan. POI: Multiple Object Tracking with
High Performance Detection and Appearance Feature. In BMTT, SenseTime Group
Limited, 2016.
```

## Running the tracker

[MOT16 benchmark](https://motchallenge.net/data/MOT16/)를 예시로 실행시키는 방법(영상을 받아올수 있는 좋은 사이트입니다.)

```
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```

- `python deep_sort_app.py -h` 이용가능한 옵션을 체크하고 저장소안에는 평가하기,시각화하기,비디오 생성하는 코드도 있다.

## Generating detections

메인 추적 응용 프로그램과 함께 사람 인식을 위한 기능을 생성성하는 스크립트가 포함되어있다. cosine similarity를 사용하여 보행자 bounding box에 시각적 모양을 비교하는데 적합하다. 아래 예제는 표준 MOTChallenge 탐지에서 이러한 특징을 생성한다.

```
python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=./MOT16/train \
    --output_dir=./resources/detections/MOT16_train
```
그 모델은 Tensorflow 1.5에서 생성되었고 호환이 되지 않는다면 pb파일로 변환해라

```
python tools/freeze_model.py
```

`generate_detections.py`는 각 시퀀스에 대해 numpy 형식의 binary file을 저장한다. 각 파일은 `Nx138`의 배열을 포함한다. 여기서 N은 해당 시퀀스의 탐지된 객체수이며 이 배열의 처음 10개 열에는 원시 MOT 검출 입력 파일에서 복사된다. 나머지 128열은 appearance descriptor를 저장한다. 이 명령으로 생성된 파일은 `deep_sort_app`에 대한 입력으로 사용이 될 수 있다.

**NOTE** : 만약 ``python tools/generate_detections.py``에서 TensorFlow에러가 발생한다면 `--model`인수에 절대경로를 전달하라

## Training the model

깊이 연관된 metric model을 학습하기 위해 [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) 접근 방식을 사용

## Highlevel overview of source files

최상위 디렉토리에는 추적 프로그램을 실행, 평가 시각화 할수 있는 실행가능 스크립트가 있다. 주된 point는 `deep_sort_app.py`에 있다. MOTChallenge 시퀀스에서 추적 프로그램 실행

* `detection.py`: 탐지 기본 클래스.
* `kalman_filter.py`: kalman filter 구현, 영상 이미지 공간 필터링을위한 구체적인 매개 변수화.
* `linear_assignment.py`: 이 모듈에는 최소 비용으로 매칭하고 매칭 cascade가 포함되어있다.
* `iou_matching.py`: 이 모듈에는IOU matching metric가 포함되어있다.
* `nn_matching.py`: 이 모듈에는nearest neighbor matching metric가 포함되어있다.
* `track.py`: 트랙 클래스는 kalman 같은 single-target track 데이터를 포함한다. state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: 이것은 다중 타겟 클래스다.

* `deep_sort_app.py`는 `.npy`에 저장된 사용자 정의 형식의 detection을 기대한다. 이들은 MOTChallenge 탐지를 사용해 계산할수 있다.

* `generate_detections.py`. 또한 미리 학습된 탐지를 제공한다. -> [pre-generated detections](https://drive.google.com/open?id=1VVqtL0klSUvLnmBKS89il1EKC3IxUBVK).

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
