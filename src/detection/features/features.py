from detection.features.age_and_gender import detect_age_and_gender
from utils.conf import FEATURES
from detection.features.clothes import detect_top_bottom_clothes

def features_prediction(frame):
    features = []
    data = []
    data = detect_age_and_gender(frame)
    clothes = detect_top_bottom_clothes(frame)
    features.append(data)
    return features
