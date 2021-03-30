from detection.yolov3 import darknet
from detection.yolov3.yolo import YOLO
#from src.utils.conf import yolo_weights

class YOLOV3(YOLO):
    def __init__(self, meta_path, cfg_path, weights_path):
        super().__init__()
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
