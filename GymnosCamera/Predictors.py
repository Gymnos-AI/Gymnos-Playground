# Continuously capture frames and perform object detection on them
import cv2


class Predictors:
    def __init__(self, model_type):
        if model_type == 'HOG':
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def hog_detector(self, to_predict):
        """
        Takes in a frame, runs it through the HOG Detector  and returns the results

        :param to_predict: The frame passed into the model
        :return: frame after it is processed
        """
        (rects, weights) = self.hog.detectMultiScale(to_predict, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(to_predict, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return to_predict
