from src.detection.yolov3 import darknet
from src.detection.yolov3.yolo import YOLO
from src.utils.conf import yolo_weights

class YOLOV3(YOLO):
    def __init__(self, meta_path=yolo_weights["data"], cfg_path=yolo_weights["cfg"], weights_path=yolo_weights["weights"]):
        super().__init__()
        self.names = self.load_classes("data/yolo/ericsson.names")
        self.meta_path = meta_path
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.load_weights()

    def load_weights(self):
        self.update_weights(self.meta_path, self.cfg_path, self.weights_path)

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def update_weights(self, meta_path, cfg_path, weights_path):
        self.metaMain = darknet.load_meta(meta_path.encode("ascii"))
        self.netMain = darknet.load_net_custom(cfg_path.encode(
            "ascii"), weights_path.encode("ascii"), 0, 1)  # batch size = 1
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)

    def load_yolo(self):
        meta_path = "./data/yolo/data.data"
        cfg_path = "./data/yolo/cfg.cfg"
        weights_path = "./data/yolo/yolo_weights.weights"
        self.update_weights(meta_path, cfg_path, weights_path)
        self.isEricsson, self.isRobocup = True, False

    def load_pan(self):
        meta_path = "./data/yolo/data.data"
        cfg_path = "./data/yolo/pan.cfg"
        weights_path = "./data/yolo/pan_optimized_last.weights"
        self.update_weights(meta_path, cfg_path, weights_path)
        self.isEricsson, self.isRobocup = True, False

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def draw(self, detections, img):
        for detection in detections:
            self.draw_box(detection, img, detection[0])
        return img

    def draw_box(self, detection, img, label, color=[255, 0, 0], text_color=[255, 255, 255]):
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = self.convertBack(
            float(x), float(y), float(w), float(h))

        self.plot_one_box((xmin, ymin, xmax, ymax), img, label=label, color=color, text_color=text_color)

    def detect(self, arr):
        return darknet.detect_image(self.netMain, self.metaMain, arr, thresh=0.25)

    def correctly_worn(self, arr, detections, poses, conf_threshold):
        isCasque, isHarness, isBoucle, isChaussure = False, False, False, False
        for detection in detections:
            cls = detection[0]
            conf = detection[1]
            x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(x, y, w, h)
            valid = conf > conf_threshold
            name = self.names[int(cls)]
            localized = False
            if cls == 2 and valid:
                if self.check_buckle(x, y, poses):
                    localized = True
                    isBoucle = True
                else:
                    localized = False
            elif cls == 0 and valid:
                isCasque = self.box_contains_joints(xmin, ymin, xmax, ymax, poses, ["Nose", "REar", "LEar", "LEye", "REye"])
                localized = isCasque
            elif cls == 1 and valid:
                isHarness = True
                localized = isHarness
            elif cls == 3 and valid:
                left = self.box_contains_joints(xmin, ymin, xmax, ymax, poses, ["LBigToe", "LSmallToe", "LHeel"])
                right = self.box_contains_joints(xmin, ymin, xmax, ymax, poses, ["RBigToe", "RSmallToe", "RHeel"])
                isChaussure = left or right
                localized = isChaussure
            label = '%s %.2f' % (name, detection[1])
            color = self.choose_color(detection[1], conf_threshold, localized=localized)
            text_color = [0,0,0] if color == [0, 255, 0] else [255, 255, 255]
            self.draw_box(detection, arr, label, color=color, text_color=text_color)
        return isCasque, isHarness, isBoucle, isChaussure
