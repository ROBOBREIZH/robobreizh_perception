echo  "Which GPU do you have ? (type 1 or 2)"
echo  "1. Nvidia RTX (Turing Architecture)"
echo  "2. Nvidia GTX or TITAN (Pascal Architecture)"
read Res

if [ "$Res" = "1" ]; then
	model="rtx"
elif [ "$Res" = "2" ]; then
	model="gtx"
else 
	echo "Invalid input, please try again"
	exit 1
fi

mkdir -p install/ && cd install/
sudo apt-get install python-pip
pip install gdown

################################################
###      Install CUDNN if not existing       ###
################################################
echo -e "\n \e[43m Check CUDNN version ... \e[0m \n"; 
gdown https://drive.google.com/uc?id=1mQuzRPWnNKnRtsQDoAw7X8X15GMOfd9o
sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb
sudo ldconfig
cp -r /usr/src/cudnn_samples_v8/ .
cd  ./cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN | grep "Test passed!"
if [ $? = 0 ]; then
       cd ../.. 
       echo -e "\n \e[42m CUDNN 8.1 already installed, skipping ... \e[0m \n"; 
else
       cd ../.. 
       echo -e "\n \e[43m Installing CUDNN 8.1 ... \e[0m \n"; 
       gdown https://drive.google.com/uc?id=145G84LZnHNi6QsCGX_7OAKrunSoKoKIa
       gdown https://drive.google.com/uc?id=1QjexX5lXDN4Qjf8hOfbB0iV1pjC0Yp9j
       sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
       sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
       sudo ldconfig
       echo -e "\n \e[42m Done. \e[0m \n"; 
fi
cd ..


################################################
### Download all the YOLO / OpenPose weights ###
################################################
cd ./data
echo -e "\n \e[43m Downloading YOLO / OpenPose weights ... \e[0m \n"; 
#Download weights for face recognition.
#Source: https://github.com/spmallick/learnopencv/tree/master/AgeGender
mkdir -p face && cd face
wget -nc https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel
wget -nc https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel
wget -nc https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb
wget -nc https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt
wget -nc https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt
wget -nc https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt
#Download models for openpose.
#Source: https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0
mkdir -p ../openpose && cd ../openpose
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AADX_DKAX6VjZ4x2Njdu_j5aa/body_pose_deploy.prototxt
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABc4g_SsCVeQj8wmHnkEHSWa/body_pose.caffemodel
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AACB31qQm3H_hqkzWGLWYjFIa/hand_pose_deploy.prototxt
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AAApu9PiOpzGYEUqzIzsxqbFa/hand_pose_model.pth
wget -nc https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AAC2TIWnbvaHf-MfQPxT4pf3a/hand_pose.caffemodel
#Download weights for clothes detection:
#Source: https://drive.google.com/drive/folders/1b7laIv9-oeh59XSV6aOO50eMKbTGsPoP
#Wget from google drive: https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/
cd ../yolo/clothes
wget -nc --no-check-certificate 'https://docs.google.com/uc?export=download&id=100wCUH7qu7DfgVkMpSCPRy7so2yXeY23' -O df2.names
wget -nc --no-check-certificate 'https://docs.google.com/uc?export=download&id=10aXSD_odnugZAw6tGRTqpTio3tOOgVNa' -O yolov3-df2.cfg
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw" -O yolov3-df2_15000.weights && rm -rf /tmp/cookies.txt
cd ../../..
echo -e "\n \e[42m Done. \e[0m \n"; 

################################################
###    Install python3.7 and Dependencies    ###
################################################
echo -e "\n \e[43m Installing python3.7 and Dependencies ... \e[0m \n"; 
sudo apt-get install build-essential
sudo apt-get install zlib1g-dev
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.7 -y
sudo apt-get install python3.7-dev -y
sudo apt-get install python3-pip -y
sudo python3.7 -m pip install --upgrade pip setuptools wheel
sudo python3.7 -m pip install -r dependencies/python_dependencies/requirements.txt
sudo python3.7 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
echo -e "\n \e[42m Done. \e[0m \n"; 

################################################
###      Download and Install Mask-RCNN      ###
################################################
# From: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
echo -e "\n \e[43m Download and Install Mask-RCNN ... \e[0m \n"; 
python3.7 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
echo -e "\n \e[42m Done. \e[0m \n";

################################################
###         Download and Install Yolo        ###
################################################
echo -e "\n \e[43m Download and Install Yolo ... \e[0m \n"; 
cd dependencies/yolo
bash install.sh $model
cd ../..
rm -rf install/
echo -e "\n \e[42m Done. \e[0m \n";

echo -e "\n \e[42mInstall complete, well done !!! \e[0m \n"; 

