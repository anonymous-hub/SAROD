
 Object Detection for SAR images

Yolov3: https://github.com/eriklindernoren/PyTorch-YOLOv3
Yolov5: https://github.com/ultralytics/yolov5
EfficientObjectDetection: https://github.com/uzkent/EfficientObjectDetection
mmdetection: https://github.com/open-mmlab/mmdetection


# SAROD: Efficient End-to-end Object Detection on SAR Images withReinforcement Learning
Anonymous WACV submission

## Overview of our framework.
<img src='./image/overview.png' width=1000>


## Clone
```
git clone https://github.com/anonymous-hub/SAROD
```

## Dataset
HRSID Dataset can be downloaded in [here](https://github.com/chaozhong2010/HRSID)

Pre-Processed dataset for the result can be downloaded by running the file or [TBA](TBA).

A example script for downloading the testset is as follows:
```
# Download the dataset
cd dataset
bash download_HRSID_cropped.sh
cd ..
```

## Download pre-trained model weights
The pretrained weights can be downloaded by running the files or [TBA](TBA).

```
# Download the pre-trained weights
cd weights
bash download_SAROD_RL_weights.sh
bash download_yolov5_480_weights.sh
bash download_yolov5_96_weights.sh
bash download_yolov3_480_weights.sh
bash download_yolov3_96_weights.sh
bash download_retinanet_weights.sh
bash download_faster_rcnn_weights.sh
cd ..
```

## Setup
```
pip install -r requirements.txt
```

