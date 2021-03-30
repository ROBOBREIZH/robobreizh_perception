# RoboBreizh Perception Package

## Overview

Robobreizh's perception package. It performs object and person detections using mask-RCNN and OpenPose / YOLO.

It can detect:

* Bags (mask-RCNN)
* Chairs (mask-RCNN)
* People (mask-RCNN)
* A person's age/gender (Based on: https://github.com/spmallick/learnopencv/tree/master/AgeGender)
* Clothes (YOLO, Based on: https://github.com/simaiden/Clothing-Detection)

Chairs are further divided into two subgroups: taken and empty. 

## Prerequisites

This package is currently only working on Ubuntu 16.04 and on computer with a NVIDIA graphic card.

### 1. Install NVIDIA Cuda 11.2

This installation of Cuda is compatible with most of the RTX / GTX / TITAN architecture, if you have another GPU please check the compatibility and download the required version at [Nvidia CUDA Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)..
Execute this script and follow instruction to install the NVIDIA Driver 11.2.

```buildoutcfg
install_cuda.sh 
```

Then reboot your computer to finish installation.

```buildoutcfg
sudo reboot 
```


### 2. Other Dependencies (CUDNN, MaskRCNN, YOLO and OpenPose)

Install dependencies with install.sh.

```buildoutcfg
bash install.sh
```

The installation script include for CUDNN, mask-RCNN, yolo (darknet, used for clothing detection), weights, python 3.7 and python dependencies.

If you encountered problems with CUDNN install you can folow the official tutorials by NVIDIA:

More information on the [Nvidia CUDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

[OPTIONAL] The dependencies can also be installed individually with:

```buildoutcfg
cd dependencies/install/{NAME_OF_DEPENDENCY}
bash install.sh
```

## Webcam/Camera demo

A webcam/camera demo is available:

```
cd src/robobreizh_perception/src
python3 video_demo.py
```

It will perform pose estimation and object detection by using the video feed from a camera connected to the computer.

## Start the server

The server is started as follow:

```buildoutcfg
cd src/robobreizh_perception/src
sudo python3.7 start_server.py
```

After a few seconds, the different weights will be loaded and the server will be ready to take requests.

Once the server has started, open a second a terminal and enter:

```buildoutcfg
python3.7 test.py
```

This will send two images (file table.png and waving-hand.jpg) to the server. 

The predictions will appear in the terminal and images will be saved at robobreizh_perception/src/demo.

## Structure

```buildoutcfg
|data: Contains the trained weights for yolo/openpose.
|dependencies: Scripts to install all the dependencies.
|src:
|---detection: yolo/openpose/mask-rcnn python implementations.
|---data: Store the images from tests.
|---utils: Utility files to read cfg.yaml and base64 conversions.
```

## FAQ

### The server can't find the weights.

The weights may not have been downloaded. 

```buildoutcfg
cd dependencies/install/data
bash install.sh
```

If it hasn't worked, check whether the weights were correctly downloaded in data.
