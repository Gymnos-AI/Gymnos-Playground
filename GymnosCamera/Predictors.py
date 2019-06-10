# Continuously capture frames and perform object detection on them
import cv2
import YoloNetwork.YoloV3 as yolo
import numpy as np

class Predictors:
    def __init__(self, model_type):
        if model_type == 'HOG':
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif model_type == "YOLOV3":
            self.model = yolo.YOLO()

    def hog_detector(self, to_predict):
        """
        Takes in a frame, runs it through the HOG Detector  and returns the results

        :param to_predict: The frame passed into the model
        :return: frame after it is processed
        """
        (rects, weights) = self.model.detectMultiScale(to_predict, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(to_predict, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return to_predict

    def yolo_v3_detector(self, to_predict):
        return np.asarray(self.model.detect_image(to_predict))
