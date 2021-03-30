import yaml
import cv2
import os

libdarknet_path = os.environ['HOME']+"/darknet/libdarknet.so"

from detection.openpose.openpose_pytorch import OpenPosePytorch
OPENPOSE = OpenPosePytorch().getInstance()

from detection.yolov3.yolov3 import YOLOV3
YOLO = None

class Features:
    def __init__(self):
        self.faceProto="../data/face/opencv_face_detector.pbtxt"
        self.faceModel="../data/face/opencv_face_detector_uint8.pb"
        self.ageProto="../data/face/age_deploy.prototxt"
        self.ageModel="../data/face/age_net.caffemodel"
        self.genderProto="../data/face/gender_deploy.prototxt"
        self.genderModel="../data/face/gender_net.caffemodel"
        self.MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        self.ageList=['1', '5', '10', '18', '25', '35', '45', '60']
        self.genderList=['M','F']
        self.hairList=["straight", "wavy", "curly", "kinky", "braids", "dreadlocks", "short"]

        self.faceNet=cv2.dnn.readNet(self.faceModel,self.faceProto)
        self.ageNet=cv2.dnn.readNet(self.ageModel,self.ageProto)
        self.genderNet=cv2.dnn.readNet(self.genderModel,self.genderProto)

        self.clothesWeights = "../data/yolo/clothes/yolov3-df2_15000.weights"
        self.clothesCfg = "../data/yolo/clothes/yolov3-df2.cfg"
        self.clothesNames = "../data/yolo/clothes/df2.names"

FEATURES = Features()

YOLO_CLOTHES = YOLOV3(meta_path="../data/yolo/clothes/df2.data", cfg_path="../data/yolo/clothes/yolov3-df2.cfg", weights_path="../data/yolo/clothes/yolov3-df2_15000.weights")

