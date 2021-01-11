import time
import cv2

from src.utils.conf import OPENPOSE
from src.detection.yolov3.letterbox import letterbox
from src.detection.features.features import features_prediction
from src.detection.maskrcnn import maskrcnn

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
        #    dest = './data/demo/video.mp4'
        #    return self.detect_video(arr, dest)
        #else:
        return self.detect_image(arr)

    def detect_image(self, arr):
        '''
        Perform computer vision detection on the image based on the argument (with or without movement, features detection...)
        '''
        t = time.time()
        features = []
        arr_bags, arr_hands, arr_persons, arr_empty_chairs, arr_taken_chairs = [], [], [], [], []
        #Perform pose estimation.
        if self.withMovement:
            poses = OPENPOSE.predict(arr)
            print('Openpose detection. (%.3fs)' % (time.time() - t))
            if self.displayPose: OPENPOSE.draw(arr, poses)
            arr_hands = self.has_raising_hands(poses)

        #Perform detection with mask RCNN for certain objects.
        if not self.onlyMovement:
            out, image = maskrcnn.mask_rcnn(arr)
            arr_persons = maskrcnn.get_masks(out, [0])
            arr_bags = maskrcnn.get_masks(out, [24, 26, 28])
            arr_empty_chairs, arr_taken_chairs = self.chairs_prediction(out)

        #Features detection include age, gender and clothes detection.
        if self.features:
            features = features_prediction(arr)
            print("Features: ", features)

        arr = image
        return arr, arr_hands, arr_bags, arr_persons, arr_empty_chairs, arr_taken_chairs, features

    def resize(self, frame, shape):
        '''

        Resize the video to improve performance if the video is too large.

        :param frame:
        :param shape:
        :return:
        '''
        res = letterbox(frame, new_shape=shape)[0]
        return res

    def chairs_prediction(self, out):
        '''
        Return 2 lists including the chairs that are taken and the ones that are still available using mask-RCNN.
        :param out:
        :return:
        '''
        arr_empty_chairs, arr_taken_chairs = [], []
        instances = out["instances"]
        classes = instances.pred_classes
        boxes = instances.pred_boxes

        #Check whether the bouding box of chairs overlap with persons.
        for i in range(0, len(classes)):
            if classes[i] == 56:
                empty = True
                for j in range(0, len(classes)):
                    if classes[j] == 0:
                        if self.intersecting(boxes[i], boxes[j]):
                            empty = False
                # Add the mask to the appropriate list
                mask = instances.pred_masks[i].cpu().tolist()
                if empty:
                    arr_empty_chairs.append(mask)
                else:
                    arr_taken_chairs.append(mask)
        return arr_empty_chairs, arr_taken_chairs

    def has_raising_hands(self, poses):
        '''
        Check whether a person is raising his hand based on the position of his shoulders and wrists/elbow.
        '''
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
        return arr_hands

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
        '''
        Check that neither range is completely greater than the other (no overlapping).
        '''
        return (a_min <= b_max) and (b_min <= a_max)

    def cleaning(self, arr_detections):
        '''
        Remove duplicate detections.
        '''
        arr_cleaned = arr_detections

        for detection in arr_detections:
            for other in arr_detections:
                if self.intersecting(detection, other) and detection != other:
                    arr_cleaned.remove(detection)
        return arr_cleaned