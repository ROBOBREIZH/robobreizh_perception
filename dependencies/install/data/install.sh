cd ../../../
#Download weights for face recognition.
#Source: https://github.com/spmallick/learnopencv/tree/master/AgeGender
cd data/face
wget https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel
wget https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel
wget https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb
wget https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt
wget https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt
wget https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt
#Download models for openpose.
#Source: https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0
cd ../openpose
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AADX_DKAX6VjZ4x2Njdu_j5aa/body_pose_deploy.prototxt
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABc4g_SsCVeQj8wmHnkEHSWa/body_pose.caffemodel
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AACB31qQm3H_hqkzWGLWYjFIa/hand_pose_deploy.prototxt
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AAApu9PiOpzGYEUqzIzsxqbFa/hand_pose_model.pth
wget https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AAC2TIWnbvaHf-MfQPxT4pf3a/hand_pose.caffemodel
#Download weights for clothes detection:
#Source: https://drive.google.com/drive/folders/1b7laIv9-oeh59XSV6aOO50eMKbTGsPoP
#Wget from google drive: https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/
cd ../yolo/clothes
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=100wCUH7qu7DfgVkMpSCPRy7so2yXeY23' -O df2.names
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=10aXSD_odnugZAw6tGRTqpTio3tOOgVNa' -O yolov3-df2.cfg
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw" -O yolov3-df2_15000.weights && rm -rf /tmp/cookies.txt
