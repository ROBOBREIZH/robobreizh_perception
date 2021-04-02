from detect import DetectRobocup
import cv2

class VideoDemo:
    def __init__(self, size=608, source=0, detector=DetectRobocup(), freq=10):
        self.size = size
        self.source = source
        self.detector = detector
        self.freq = freq
        self.detect_video()

    def detect_video(self):
        cap = cv2.VideoCapture(self.source)
        d = 0
        f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            d = d + 1
            if ret:
                if (d % self.freq == 0):
                    f = f + 1
                    resized = self.detector.resize(frame, self.size)
                    res, arr_hands, arr_bags, arr_persons, arr_empty_chairs, arr_taken_chairs, features, ids = self.detector.detect_image(
                        resized)
                    msk_res = self.detector.resize(res, self.size)
                    cv2.imshow('Demo', msk_res)
                    cv2.waitKey(3)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    VideoDemo()