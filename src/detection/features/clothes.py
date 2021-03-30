import cv2
from utils.conf import FEATURES
import numpy as np

net = cv2.dnn.readNet(FEATURES.clothesWeights, FEATURES.clothesCfg)
from utils.conf import YOLO_CLOTHES

classes = None
classes = FEATURES.clothesNames

#scale = 0.00392
scale = 1.0
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.25
nms_threshold = 0.4

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def clothes_detect(image):
    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    #print("out: ", outs)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print('scores: %s, class_id: %s, confidence: %s' % (scores, class_id, confidence))
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    print("ids: ", class_ids)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    print("Indices: ", indices)
    
def detect_top_bottom_clothes(frame):
    '''
    Return the top and bottom clothes a person is wearing.
    '''
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
    return clothes
