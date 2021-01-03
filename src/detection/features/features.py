import cv2
import math
import argparse

from src.utils.conf import FEATURES
from src.detection.features.clothes import clothes_detect
from src.utils.conf import YOLO_CLOTHES

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

classes = None
with open(FEATURES.clothesNames, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def features_prediction(frame):
    #i = clothes_detect(frame)
    raw_detection = YOLO_CLOTHES.detect(frame)
    print("raw: ", raw_detection)

    clothes = ["", ""]
    no_top = True
    no_bottom = True
    for detection in raw_detection:
        cls_id = detection[0]
        if cls_id < 6 or cls_id > 8 and no_top:
            clothes[0] = classes[cls_id]
            no_top = False
        elif 6 <= cls_id <= 8 and no_bottom:
            clothes[1] = classes[cls_id]
            no_bottom = False
    print("clothes: ", clothes)

    features = []
    resultImg,faceBoxes=highlightFace(FEATURES.faceNet,frame)
    padding=20
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        data = []
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), FEATURES.MODEL_MEAN_VALUES, swapRB=False)
        FEATURES.genderNet.setInput(blob)
        genderPreds=FEATURES.genderNet.forward()
        gender=FEATURES.genderList[genderPreds[0].argmax()]
        FEATURES.ageNet.setInput(blob)
        agePreds=FEATURES.ageNet.forward()
        age=FEATURES.ageList[agePreds[0].argmax()]
        print("agePreds: ", agePreds)

        data.append(faceBox)
        data.append(int(age))
        data.append(gender)
        data.append(clothes)
        features.append(data)
    return features
