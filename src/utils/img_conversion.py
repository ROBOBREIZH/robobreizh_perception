import base64

import cv2
import numpy as np


def base64_to_cv2(encoded_data):
    """
    Convert an image in base64 format to cv2 format"

    :param encoded_data:
    :return:
    """
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_base64(dst):
    """
    Convert an image in cv2 format to base64 format"

    :param encoded_data:
    :return:
    """
    result, dst_data = cv2.imencode('.jpg', dst)
    dst_base64 = base64.b64encode(dst_data)
    return dst_base64


def cv2_to_base64_png(dst):
    """
    Convert an image in cv2 format to base64 format"

    :param encoded_data:
    :return:
    """
    dst_data = cv2.imencode('.png', dst)[1]
    dst_base64 = base64.b64encode(dst_data)
    return dst_base64



def base64_to_video(path, encoded_data):
    """
    Decode a video in base64 format and save it in path."

    :param encoded_data:
    :return:
    """
    videodata = base64.b64decode(encoded_data)
    with open(path, 'wb') as f:
        f.write(videodata)
