# PoseEstimation_Scoring-Your-Video <a href="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/Compare_pose.ipynb">[Demo]</a>

## 주요기능

AlphaPose를 backbone으로 사용하여 영상의 자세를 추정하였고, 추정된 값을 사용하여 아래의 기능을 구현하였다.

### 0. AlphaPose에서 얻은 자세 추정값을 계산에 사용하기 위해 l2 normalization 한다.

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/img/img_0.jpg?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

이미지마다 **높이와 너비가 다르고, 사람의 위치와 크기 또한 다르기때문에** 이를 고려하여 l2 normalization을 진행해야한다. 따라서 다음의 **두 가지 단계**를 통해 이를 구현하였다.

1. 검출된 사람의 바운딩 박스를 1대1 비율로 crop한다.

2. crop된 박스에서 각 부위별 좌표값으로 l2 normalization을 한다.

```
# usage

!python alphapose_compare.py --cfg /configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint /pretrained_models/fast_res50_256x192.pth --video /data/video/TKD_slow.mp4 --save_video --outdir /data/video/result

!python alphapose_compare.py --cfg /configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint /pretrained_models/fast_res50_256x192.pth --image /data/image/TKD/TKD_6.png --save_img --outdir /data/image/result

## python

import Compare_pose

Compare_pose.l2_normalize("data/video/result/alphapose-TKD_1.json")

# input json

[{"image_id": "TKD_1.png", "category_id": 1, "keypoints": [706.6857299804688, 165.55355834960938, 0.9743795990943909, 722.0609741210938, 142.49066162109375, 0.9858314990997314, ....], "score": 3.09490704536438, "box": [536.5984497070312, 60.53300857543945, 425.3385009765625, 787.2135124206543], "idx": [0.0]}]

# output json

[{"image_id": "TKD_1.png", "category_id": 1, "keypoints": [0.19470483935752733, 0.03471309514819291, 0.9743795990943909, 0.19978691437930066, 0.02708997252836812, 0.9858314990997314, ....], "score": 3.09490704536438, "box": [536.5984497070312, 60.53300857543945, 425.3385009765625, 787.2135124206543], "idx": [0.0]}]
```

### 1. 특정 자세를 취하고 있는 자신의 이미지(왼쪽)을 넣으면 타겟 동영상 전체에서 유사한 자세를 취하고 있는 프레임(오른쪽)을 찾아 출력하고 두 자세간의 유사도를 부위별로 채점하여 보여준다. 

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/result_1.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

이 code에서는 위와 같은 기능을 구현하기 위해 **세 가지 알고리즘**을 제안하였다.

**1. Cosine Similarity**

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/img/cos.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

위와 같은 식을 사용하여 **두 자세간의 cosine similarity를 구하고, 이를 통해 유사도를 계산**하여 0~100사이 값으로 점수화 하였다.

```
# usage
## python

import Compare_pose

Score = Compare_pose.cos_sim("data/image/result/alphapose-TKD_6_l2norm.json","data/video/result/alphapose-TKD_test_l2norm.json", "data/image/TKD/TKD_6.png", 'data/video/TKD_test.mp4')
```

**2. Weight Matching(l1 norm)**

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/img/wm.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

위의 식에서 F<sub>Ck</sub>는 PoseEstimation에서 추정된 부위의 위치가 얼마나 정확한지를 나타내는 값이다. 따라서 두 자세를 비교할때 **더 정확히 추론된 부위에 가중치를 주어 계산**하는 방법이다. 이를 0~100사이 값으로 점수화 하여 출력하였다.

```
# usage
## python

import Compare_pose

Score = Compare_pose.weightmatch("data/image/result/alphapose-TKD_6_l2norm.json","data/video/result/alphapose-TKD_test_l2norm.json", "data/image/TKD/TKD_6.png", 'data/video/TKD_test.mp4')
```

**3. Weight Matching(l2 norm)**

두 자세를 비교할때 **더 정확히 추론된 부위에 가중치를 주어 계산**하는 방법이다. 이를 0~100사이 값으로 점수화 하여 출력하였다. 2번의 방법과 유사하지만 **l2 norm을 사용**하였다.

```
# usage
## python

import Compare_pose

Score = Compare_pose.l2_weightmatch("data/image/result/alphapose-TKD_6_l2norm.json","data/video/result/alphapose-TKD_test_l2norm.json", "data/image/TKD/TKD_6.png", 'data/video/TKD_test.mp4')
```

### 2. 특정 동작을 하고 있는 두 영상을 비교하여 각 부위별로 유사도를 채점하여 보여준다. (두 영상의 길이와 속도에 영향을 받지 않는다.)

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/result_3.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

위의 결과는 태권도 품세중 '고려'의 일부 구간을 **11초동안 동작한 영상**과 **33초동안 동작한 영상**을 비교한 결과이다.

**기존의 방법(Euclidean)으로는** 두 동영상의 유사도를 비교할때 같은 동작이더라도 영상간의 시점이 다르거나 동작의 **실행속도가 다르면 비교할 수 없었다.** 하지만 이 code에서는 **Dynamic Time Warping을 사용하여 이러한 문제점을 해결**하였다. 

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/img/DTW.jpg?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

frame의 흐름을 x축으로 놓고 추정된 자세의 값을 y축으로 놓았을때 위와 같은 그래프가 형성된다. 이때 Dynamic Time Warping은 기존의 방법(Euclidean)과 다르게 같은 시점을 기준으로 비교하는 것이 아닌 **값의 흐름을 기준으로 비교**하여, **두 영상간의 길이가 다르더라도 비교할 수 있게 된다.**

```
# usage
## python

import Compare_pose

Score = Compare_pose.dtw_compare("data/video/result/alphapose-TKD_test_l2norm.json","data/video/result/alphapose-TKD_slow_l2norm.json")
```
## Comming Soon

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/208.png?raw=true?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

기준점수를 설정하면 해당점수보다 낮을시 스켈레톤을 붉은색으로 표시하는 기능을 개발하였고, 리팩토링 후 공개할 예정입니다. 
