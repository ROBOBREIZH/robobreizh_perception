from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
import time

from detection.maskrcnn.hacked_visualizer import HackedVisualizer

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

SAVE_IMG = True


def mask_rcnn(im):
    '''
    Perform the mask RCNN detection
    '''
    outputs = predictor(im)
    v = HackedVisualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = out.get_image()
    if SAVE_IMG:
        cv2.imwrite('./data/demo.png', image)
        #print("out: ", out.get_image())
    return outputs, image


def get_masks(outputs, classes=[0]):
    '''
    Get all the masks for a given class.
    '''
    masks = []
    instances = outputs["instances"]

    for i in range(0, len(instances.pred_classes)):
        for cls in classes:
            if instances.pred_classes[i] == cls:
                mask = instances.pred_masks[i].cpu().tolist()
                masks.append(mask)
    return masks

def get_ids(outputs):
    classes = outputs['instances'].get('pred_classes').to("cpu").numpy()
    return classes

