sudo apt update
sudo apt install libopencv-dev python3-opencv
cd dependencies/install
#Install Cuda
#bash cuda/ubuntu18/install.sh
#Download all the YOLO / OpenPose weights
cd data/
bash install.sh
cd ..
#Download python3.6
#bash python36/install.sh
#Download python dependencies
bash python_dependencies/install.sh
#Download MaskRCNN
bash maskrcnn/install.sh
#Download YOLO
cd yolo
bash install.sh
cd ..
