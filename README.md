# robobreizh_vision

## Overview

Robobreizh's computer vision server. It performs object and person detections using mask-RCNN and YOLO.

It can detect:

* Bags (mask-RCNN)
* Chairs (mask-RCNN)
* People (mask-RCNN)
* A person's age/gender (Based on: https://github.com/spmallick/learnopencv/tree/master/AgeGender)
* Clothes (YOLO, Based on: https://github.com/simaiden/Clothing-Detection)

Chairs are further divided into two subgroups: taken and empty.  

## Prerequisites

Install dependencies with install_ubuntu16.sh or install_ubuntu18.sh dependencies on the OS.

```buildoutcfg
bash install_ubuntu16.sh
```
The installation script include for CUDA, mask-RCNN, yolo (darknet, used for clothing detection), weights, python 3.6 (Ubuntu 16 only) and python dependencies.

The dependencies can also be installed individually with:

```buildoutcfg
cd dependencies/install/{NAME_OF_DEPENDENCY}
bash install.sh
```

## Start the server

In order to start the server, open a terminal and enter:

```buildoutcfg
python3 main.py
```

After a few seconds, the different weights will be loaded and the server will be ready to take requests.

## Test

Once the server has started, open a second a terminal and enter:

```buildoutcfg
python3 test_robocup.py
```

This will send an image to the server. 

The predictions will appear in the terminal and the image will be saved at data/test_client/demo.png.

## Structure

```buildoutcfg
|data: Contains the trained weights for yolo/openpose.
|dependencies: Scripts to install all the dependencies.
|src:
|---detection: yolo/openpose/mask-rcnn python implementations.
|---request: Defines the requests handled by the server and their formats.
|---server: Start the server and handles the HTTP request.   
|---utils: Utility files to read cfg.yaml and base64 conversions.
```

## FAQ

### The server can't find the weights.

The weights may not have been downloaded. 

```buildoutcfg
cd dependencies/install/data
bash install
```

If it hasn't worked, check that th