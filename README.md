# PoseEstimation_Scoring-Your-Video <a href="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/Compare_pose.ipynb">[demo]</a>

## Usage

```
import Compare_pose as i2v
```

## 주요 기능

AlphaPose를 backbone으로 사용하여 영상의 자세를 추정하였고, 추정된 값을 사용하여 아래의 기능을 구현하였다.

### 1. 특정 자세를 취하고 있는 자신의 이미지(왼쪽)을 넣으면 타겟 동영상 전체에서 유사한 자세를 취하고 있는 프레임(오른쪽)을 찾아 출력하고 두 자세간의 유사도를 부위별로 채점하여 보여준다. 

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/result_1.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

 세 가지 방법으로 두 자세간의 유사도를 계산하였다.

### 2. 특정 동작을 하고 있는 두 영상을 비교하여 각 부위별로 유사도를 채점하여 보여준다.

<img src="https://github.com/HwangToeMat/PoseEstimation_Scoring-Your-Video/blob/master/result_3.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">


또한, 실행속도가 다르지만 같은 동작인 두 영상의 유사도를 계산하기위해 Dynamic Time Warping을 사용하였다.
