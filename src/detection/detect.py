import time
import cv2

from src.utils.conf import OPENPOSE
from src.detection.yolov3.letterbox import letterbox
from src.detection.features.features import features_prediction

import torch
from src.detection.maskrcnn import maskrcnn
import random

class Detect:
    def __init__(self, displayBag=True, displayChair=True, displayPerson=True, displayPose=True, displayWaving=True, isMovement=True, onlyMovement=False, features=False, isVideo=False):
        self.displayBag = displayBag
        self.displayChair = displayChair
        self.displayPerson = displayPerson
        self.displayPose = displayPose
        self.displayWaving = displayWaving
        self.withMovement = isMovement
        self.onlyMovement = onlyMovement
        self.features = features
        self.last_picture = None

    def detect(self, arr):
        # Video is was done just for the purpose of making a short video and will lead to a crash."
        #if self.isVideo:
        #    dest = '/home/cerv/rtsd/data/demo/video.mp4'
        #    return self.detect_video(arr, dest)
        #else:
        return self.detect_image(arr)

    def detect_image(self, arr):
        t = time.time()
        features = []
        arr_bags, arr_hands, arr_persons, arr_empty_chairs, arr_taken_chairs = [], [], [], [], []
        if self.withMovement:
            poses = OPENPOSE.predict(arr)
            print('Openpose detection. (%.3fs)' % (time.time() - t))
            if self.displayPose: OPENPOSE.draw(arr, poses)
            arr_hands = self.has_raising_hands(arr, poses)

        if not self.onlyMovement:
            out, image = maskrcnn.mask_rcnn(arr)
            arr_persons = maskrcnn.get_masks(out, [0])
            arr_bags = maskrcnn.get_masks(out, [24, 26, 28])
            arr_empty_chairs = maskrcnn.get_masks(out, [56])

            #arr_empty_chairs, arr_taken_chairs = self.chairs_prediction(out)

        if self.features:
            features = features_prediction(arr)
            print("Features: ", features)

        arr = image
        cv2.imwrite('/home/cerv/rtsd/data/demo/demo_test.png', arr)
        return arr, arr_hands, arr_bags, arr_persons, arr_empty_chairs, arr_taken_chairs, features

    def resize(self, frame, shape):
        """
        Resize the video to improve performance if the video is too large.

        :param frame:
        :param shape:
        :return:
        """
        res = letterbox(frame, new_shape=shape)[0]
        return res

    def chairs_prediction(self, out):
        arr_empty_chairs, arr_taken_chairs = [], []
        classes = out["instances"].pred_classes
        boxes = out["instances"].pred_boxes
        print("boxes: ", boxes)
        for i in range(0, len(classes)):
            print("cls: ", classes[i])
            if classes[i] == 56:
                for j in range(0, len(classes)):
                    if classes[j] == 0:
                        print(boxes[i])
                        print(boxes[j])
                        if self.intersecting(boxes[i], boxes[j]):
                            isEmpty = False
                            print("isEmpty: ", isEmpty)
        return arr_empty_chairs, arr_taken_chairs


    def has_raising_hands(self, arr, poses):
        arr_hands = []
        limbs = ["LWrist", "LElbow", "RWrist", "RElbow"]
        shoulders = ["LShoulder", "RShoulder"]
        for pose in poses:
            for limb in limbs:
                if pose[limb][1] != 0.0:
                    for shoulder in shoulders:
                        if pose[limb][1] < pose[shoulder][1]:
                            xmin = min(pose[limb][0], pose[shoulder][0])
                            xmax = max(pose[limb][0], pose[shoulder][0])
                            pos = (xmin, pose[limb][1], xmax, pose[shoulder][1])
                            arr_hands.append(pos)
                            #if self.displayWaving:
                            #    plot_one_box((pose[limb][0], pose[limb][1], pose[limb][0], pose[limb][1]), arr, color=[196, 196, 0], label="Waving")
                            #break
        #self.cleaning(arr_hands)
        print("hands: ", arr_hands)
        return arr_hands

    def has_bags(self, arr, detections):
        arr_bags = []
        for detection in detections:
            cls = detection[0]
            if cls == 24 or cls == 26:
                if self.displayBag:
                    YOLO.draw_box(detection, arr, "Bag " + str(round(detection[1], 2)), color=[0, 255, 0])
                arr_bags.append(self.xywh_to_xyxy(detection[2]))
        self.cleaning(arr_bags)
        print("bags: ", arr_bags)
        return arr_bags


    def has_chairs(self, arr, detections):
        arr_empty_chairs = []
        arr_taken_chairs = []
        arr_persons = []
        for detection in detections:
            isEmpty = True
            cls = detection[0]
            rect = self.xywh_to_xyxy(detection[2])
            if cls == 0 and self.displayPerson:
                YOLO.draw_box(detection, arr, "Person " + str(round(detection[1], 2)),
                              color=[255, 0, 255])
                arr_persons.append(self.xywh_to_xyxy(detection[2]))
            if cls == 56:
                for det in detections:
                    if det[0] == 0:
                        rect_person = self.xywh_to_xyxy(det[2])
                        if self.intersecting(rect, rect_person):
                            isEmpty = False
                empty = " (empty) " if isEmpty else " (taken) "
                if isEmpty:
                    arr_empty_chairs.append(self.xywh_to_xyxy(detection[2]))
                else:
                    arr_taken_chairs.append(self.xywh_to_xyxy(detection[2]))
                if self.displayChair:
                    YOLO.draw_box(detection, arr, "Chair" + empty + str(round(detection[1], 2)), color=[255, 0, 0])
        self.cleaning(arr_empty_chairs)
        self.cleaning(arr_taken_chairs)
        print("empty_chairs: ", arr_empty_chairs)
        return arr_persons, arr_empty_chairs, arr_taken_chairs


    def xywh_to_xyxy(self, xywh):
        x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
        return YOLO.convertBack(x, y, w, h)

    def intersecting(self, rect1, rect2):
        '''
        Returns whether two rectangles overlap.
        rect: [xmin, ymin, xmax, ymax]

        :param rect1:
        :param rect2:
        :return:
        '''
        x1min, y1min, x1max, y1max = rect1[0], rect1[1], rect1[2], rect1[3]
        x2min, y2min, x2max, y2max = rect2[0], rect2[1], rect2[2], rect2[3]
        return self.range_overlap(x1min, x1max, x2min, x2max) and self.range_overlap(y1min, y1max, y2min, y2max)

    def range_overlap(self, a_min, a_max, b_min, b_max):
        '''Neither range is completely greater than the other
        '''
        return (a_min <= b_max) and (b_min <= a_max)

    def cleaning(self, arr_detections):
        arr_cleaned = arr_detections

        for detection in arr_detections:
            for other in arr_detections:
                if self.intersecting(detection, other) and detection != other:
                    arr_cleaned.remove(detection)
        return arr_cleaned