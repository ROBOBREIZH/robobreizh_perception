import time
import cv2

from utils.conf import OPENPOSE
#from src.utils.conf import YOLO
from detection.yolov3.letterbox import letterbox
from detection.features.features import features_prediction

#from src.detection.yolov3.yolo import plot_one_box
import torch
from detection.maskrcnn import maskrcnn
import random

class DetectRobocup:
    def __init__(self, displayBag=True, displayChair=True, displayPerson=True, displayPose=True, displayWaving=True, isMovement=True, onlyMovement=False, features=False, isVideo=False):
        self.displayBag = displayBag
        self.displayChair = displayChair
        self.displayPerson = displayPerson
        self.displayPose = displayPose
        self.displayWaving = True
        self.isMovement = isMovement
        self.onlyMovement = onlyMovement
        self.features = features
        self.isVideo = False
        self.last_picture = None
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    def detect(self, arr):
        if self.isVideo:
            #Video is was done just for the purpose of making a short video and will lead to a crash."
            dest = '/home/cerv/rtsd/data/demo/video.mp4'
            return self.detect_video(arr, dest)
        else:
            return self.detect_image(arr)

    def detect_waving_hands(self, image, saveImage=False):
        t = time.time()

        poses = OPENPOSE.predict(image)
        print('Openpose detection. (%.3fs)' % (time.time() - t))
        if self.displayPose: OPENPOSE.draw(image, poses)
        arr_hands = self.has_raising_hands(image, poses)
        if saveImage:
            return arr_hands, image
        #cv2.imwrite('demo/demo_waving_hand.png', image)
        else:
            return arr_hands

    def detect_empty_chairs(self, image, saveImage=False):
        t = time.time()

        out, image = maskrcnn.mask_rcnn(image)
        empty_chairs, taken_chairs = self.chairs_prediction(out)
        print('Empty chairs detection. (%.3fs)' % (time.time() - t))
        if saveImage:
            arr_persons = maskrcnn.get_boxes(out, [0])
            arr_chairs = maskrcnn.get_masks(out, [56])
            self.chairs_maskrcnn_prediction(image, arr_persons, arr_chairs)
            return empty_chairs, image
        else:
            return empty_chairs

    def detect_taken_chairs(self, image, saveImage=False):
        t = time.time()

        out, image = maskrcnn.mask_rcnn(image)
        empty_chairs, taken_chairs = self.chairs_prediction(out)
        print('Taken chairs detection. (%.3fs)' % (time.time() - t))
        if saveImage:
            arr_persons = maskrcnn.get_boxes(out, [0])
            arr_chairs = maskrcnn.get_masks(out, [56])
            self.chairs_maskrcnn_prediction(image, arr_persons, arr_chairs)
            return taken_chairs, image
        else:
            return taken_chairs

    def detect_chairs(self, image, saveImage=False):        
        t = time.time()

        out, image = maskrcnn.mask_rcnn(image)
        empty_chairs, taken_chairs = self.chairs_prediction(out)
        print('Chairs (Empty/Taken) detection. (%.3fs)' % (time.time() - t))
        if saveImage:
            arr_persons = maskrcnn.get_boxes(out, [0])
            arr_chairs = maskrcnn.get_masks(out, [56])
            self.chairs_maskrcnn_prediction(image, arr_persons, arr_chairs)
            return empty_chairs, taken_chairs, image
        else:
            return empty_chairs, taken_chairs

    def detect_objects(self, image, objects_list, saveImage=False):
        t = time.time()

        res = {}
        out, img = maskrcnn.mask_rcnn(image)
        detected_obj = objects_list
        if objects_list[0] == "all":
            detected_obj = []
            ids = maskrcnn.get_ids(out)
            j=0
            for i in ids:
                j += 1
                detected_obj.append(self.class_names[i])
        print(detected_obj)
        for obj in detected_obj:
            res[obj] = maskrcnn.get_masks(out, [self.class_names.index(obj)])

        print('Objects/Person detection. (%.3fs)' % (time.time() - t))
        
        cv2.imwrite('demo/demo.png', img)
        if saveImage:
            return res, img
        else:
            return res

    def detect_features(self, image):
        t = time.time()

        features = features_prediction(image)
        print('Features detection. (%.3fs)' % (time.time() - t))

        return features

    def detect_image(self, arr):
        t = time.time()
        features = []
        arr_chairs = []
        arr_bags, arr_hands, arr_persons, arr_empty_chairs, arr_taken_chairs = [], [], [], [], []

        if self.isMovement:
            poses = OPENPOSE.predict(arr)
            print('Openpose detection. (%.3fs)' % (time.time() - t))
            if self.displayPose: OPENPOSE.draw(arr, poses)
            arr_hands = self.has_raising_hands(arr, poses)
        
        if not self.onlyMovement:
            #old code to use yolo"
            #detections = YOLO.detect(arr)
            #print('Yolo detection. (%.3fs)' % (time.time() - t))
            #arr_bags = self.has_bags(arr, detections)
            #arr_persons, arr_empty_chairs, arr_taken_chairs = self.has_chairs(arr, detections)
            out, image = maskrcnn.mask_rcnn(arr)
            ids = maskrcnn.get_ids(out)
            #self.last_picture = image
            arr_persons = maskrcnn.get_masks(out, [0])
            arr_bags = maskrcnn.get_masks(out, [24, 26, 28])
            arr_chairs = maskrcnn.get_masks(out, [56])
            #arr_boxes_chairs = maskrcnn.get_boxes(out,  [56])
            #arr_persons = maskrcnn.get_boxes(out, [0])
            #print('chairs: ', arr_chairs)
            #print('chairs[0][0][0]: ', arr_chairs[0][0][0])
            #self.chairs_maskrcnn_prediction(image, arr_persons, arr_chairs)
            #arr_empty_chairs, arr_taken_chairs = self.chairs_prediction(out)

        if self.features:
            features = features_prediction(arr)
            print("Features: ", features)

        arr = image
        cv2.imwrite('/home/cerv/rtsd/data/demo/demo_test.png', arr)
        return arr, arr_hands, arr_bags, arr_persons, arr_empty_chairs, arr_taken_chairs, features, ids

    def detect_image_all(self, arr):
        t = time.time()
     
        out, image = maskrcnn.mask_rcnn(arr)
        ids = maskrcnn.get_ids(out)
        masks_all = maskrcnn.get_all_masks(out)

        arr = image
        cv2.imwrite('/home/cerv/rtsd/data/demo/demo_test.png', arr)
        return arr, masks_all


    def resize(self, frame, shape):
        """
        Resize the video to improve performance if the video is too large.

        :param frame:
        :param shape:
        :return:
        """
        res = letterbox(frame, new_shape=shape)[0]
        return res

    def detect_video(self, dest, freq=10):
        #print("source: ", "/home/cerv/rtsd/data/demo/test.webm")
        shape = 608
        cap = cv2.VideoCapture("/home/cerv/rtsd/data/output.avi")
        #cap = cv2.VideoCapture(source)
        #cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        res = self.resize(frame, shape)
        msk_res = res
        out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (res.shape[1], res.shape[0]))
        d = 0
        f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            d = d + 1
            if ret:
                if (d % freq == 0):
                    f = f + 1
                    resized = self.resize(frame, shape)
                    res, arr_hands, arr_bags, arr_persons, arr_empty_chairs, arr_taken_chairs, features, ids = self.detect_image(resized)
                    msk_res = self.resize(res, shape)
                    out.write(msk_res)
                    cv2.imshow('Demo', msk_res)
                    cv2.waitKey(3)
                out.write(msk_res)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def chairs_maskrcnn_prediction(self, image, persons, chairs):
        chair_is_empty = True
        chair_found_start = False
        i_start, j_start = 0, 0
        height = len(chairs[0])
        width = len(chairs[0][0])
        for chair in chairs:
            for person in persons:
                for i in range(0, height):
                    for j in range(0, width):
                        if chair[i][j] and not chair_found_start:
                            chair_found_start = False
                            i_start, j_start = i, j
                            if person[i][j] and chair_is_empty:
                                chair_is_empty = False
                                self.plot_one_box((i, j, i, j), image, color=[196, 196, 0], label="Chair is taken")
            if chair_is_empty:
                self.plot_one_box((i_start, j_start, i_start, j_start), image, color=[0, 112, 225], label="Chair is empty")
            chair_is_empty = True
            chair_found_start = False


    def chairs_prediction(self, out):
        arr_empty_chairs, arr_taken_chairs = [], []
        classes = out["instances"].pred_classes
        boxes = out["instances"].pred_boxes
        print("boxes: ", boxes)
        isEmpty = True
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
                            arr_taken_chairs.append(boxes[i])
                        else:
                            arr_empty_chairs.append(boxes[i])

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
                            if self.displayWaving:
                                self.plot_one_box((pose[limb][0], pose[limb][1], pose[limb][0], pose[limb][1]), arr, color=[196, 196, 0], label="Waving")
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

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None, text_color=[255, 255, 255]):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            #print(text_color)
            cv2.putText(img=img, text=label, org=(c1[0], c1[1] - 2), fontFace=0, fontScale=tl / 3, color=text_color, thickness=tf, lineType=cv2.LINE_AA)