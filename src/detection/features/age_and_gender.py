import cv2
from utils.conf import FEATURES


def highlight_face(net, frame, conf_threshold=0.7):
    '''
    Highlight the faces to improve prediction.
    '''
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence=detections[0, 0, i, 2]
        if confidence>conf_threshold:
            x1=int(detections[0, 0, i, 3]*frame_width)
            y1=int(detections[0, 0, i, 4]*frame_height)
            x2=int(detections[0, 0, i, 5]*frame_width)
            y2=int(detections[0, 0, i, 6]*frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame_opencv_dnn,face_boxes


def detect_age_and_gender(frame):
    '''
    Detect a person's age and gender. Return a list containing face bounding box, the age and gender.
    '''
    features = []
    resultImg, faceBoxes = highlight_face(FEATURES.faceNet, frame)
    padding = 20
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        data = []
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), FEATURES.MODEL_MEAN_VALUES, swapRB=False)
        FEATURES.genderNet.setInput(blob)
        genderPreds = FEATURES.genderNet.forward()
        gender = FEATURES.genderList[genderPreds[0].argmax()]
        FEATURES.ageNet.setInput(blob)
        agePreds = FEATURES.ageNet.forward()
        age = FEATURES.ageList[agePreds[0].argmax()]
        print("agePreds: ", agePreds)

        data.append(faceBox)
        data.append(int(age))
        data.append(gender)
    return data



