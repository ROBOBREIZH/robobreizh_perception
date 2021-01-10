# robobreizh_vision

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

After a few seconds, the different weights will be loaded and the server will be ready.

## Test

Once the server has started, open a second a terminal and enter:

```buildoutcfg
python3 test_robocup.py
```

An image will be sent to the server. Object and person detections will be performed


"data/demo/demo_test.png"