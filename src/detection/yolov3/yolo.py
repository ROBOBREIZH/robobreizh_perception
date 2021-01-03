import math
import random
import cv2

class YOLO():
    def __init__(self):
        super().__init__()

    def dist_points(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def dist_arr(self, arr1, arr2):
        return math.sqrt((arr1[0] - arr2[0]) ** 2 + (arr1[1] - arr2[1]) ** 2)

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

    def choose_color(self, confidence, conf_threshold, localized=True):
        #Color follows the BGR format
        if confidence > conf_threshold:
            color = [0, 255, 0]
        if not localized:
            color = [255, 0, 0]
        if confidence < conf_threshold:
            color = [0, 0, 255]
        return color
