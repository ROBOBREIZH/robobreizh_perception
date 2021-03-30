from detection.openpose.src.body import Body

import cv2

class OpenPosePytorch():
    __instance = None

    @staticmethod
    def getInstance():
        if OpenPosePytorch.__instance == None:
            OpenPosePytorch()
        return OpenPosePytorch.__instance

    def __init__(self):
        if OpenPosePytorch.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            OpenPosePytorch.__instance = self
            self.body_estimation = Body('../data/openpose/body_pose_model.pth')

    def predict(self, img):
        candidate, subset = self.body_estimation(img)
        pred = self.get_key_limbs_position(candidate, subset)
        return pred

    def get_key_limbs_position(self, candidate, subset):
        arr = []
        for n in range(len(subset)):
            dic = {}
            if (len(subset) == 0):
                return dic
            names = ["Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "LHip"]
            ids = [1, 2, 3, 4, 5, 6, 7, 8, 11]
            for name, id in zip(names, ids):
                index = int(subset[n][id])
                if index == -1:
                    x, y = [0, 0]
                else:
                    x, y = candidate[index][0:2]
                dic[name] = [x, y]
            arr.append(dic)
        return arr

    def draw(self, img, arr):
        for dic in arr:
            for name in dic:
                x, y = dic[name]
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 6, [255, 255, 255], thickness=-1)
