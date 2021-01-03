import sys
import cv2
from src.utils.conf import openpose_path

try:
    sys.path.append(openpose_path + "/build/python")
    from openpose import pyopenpose as op
except ImportError as e:
    raise e

params = dict()
params["model_folder"] = openpose_path + "/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
# Process Image
datum = op.Datum()

class OpenPose():
    __instance = None
    @staticmethod
    def getInstance():
        if OpenPose.__instance == None:
            OpenPose()
        return OpenPose.__instance
    def __init__(self):
        if OpenPose.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            OpenPose.__instance = self

    def predict(self, img):
        """
        Perform detection on an image in cv2 arr format.
        :param img:
        :return:
        """
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        pred = self.get_key_limbs_position(datum.poseKeypoints)
        return pred

    def get_key_limbs_position(self, candidate):
        """
        Return an array with all the limbs detected

        :param candidate:
        :return: Array with all the persons detected and the position of their body limbs/
        [ {'Neck': [0, 0], "Nose": [1, 1]}, [{'LEar': [2, 2], "Rear": [3, 3]}] ]
        """


        poses = []
        #print("candidate: ", candidate)
        #print("type: ", len(candidate.shape))
        names = ["Nose", "Neck", "RShoulder","RElbow", "RWrist",
                 "LShoulder", "LElbow",  "LWrist", "MidHip", "RHip",
                 "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                 "REye", "LEye", "REar", "LEar", "LBigToe",
                 "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
        if len(candidate.shape) == 0:
            return poses
        for n in range(len(candidate)):
            dic = {}
            if (len(candidate) == 0):
                return dic
            for i in range (0, len(names)):
                x = candidate[n][i][0]
                y = candidate[n][i][1]
                dic[names[i]] = [x, y]
            poses.append(dic)
        return poses

    def draw(self, img, poses):
        """
        Draw all the body limbs detected on an image.
        :param img:
        :param poses:
        :return:
        """
        for dic in poses:
            for name in dic:
                x, y = dic[name]
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 6, [0, 255, 0], thickness=-1)
