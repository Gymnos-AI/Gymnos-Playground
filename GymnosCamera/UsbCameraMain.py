# Continuously capture frames and perform object detection on them
import cv2
import Predictors
import numpy as np


class UsbCameraMain:
    def __init__(self):
        # initialize the camera
        self.camera = cv2.VideoCapture(0)
        self.camera_height = 256
        self.camera_width = 256

        # initialize the Predictors
        self.predictor = Predictors.Predictors('YOLOV3')

    def run_loop(self):
        """
        This main loop will grab frames the camera and print it onto the screen
        """
        # capture frames from the camera
        count = 0
        # list_of_coords = [(0, 0, 0, 0)]
        while True:
            ret, image = self.camera.read()
            image = cv2.resize(image, (self.camera_height, self.camera_width))

            # run predictions every 10 frames to help keep up with live stream pace
            # if count % 10 == 0:
            list_of_coords = self.predictor.yolo_v3_detector(image)
            for (topX, leftY, bottomX, rightY) in list_of_coords:
                cv2.rectangle(image, (topX, leftY), (bottomX, rightY), (0, 0, 255), 2)

            image = np.asarray(image)
            image = cv2.resize(image, (416, 416))
            cv2.imshow("Video Feed", image)

            count = count + 1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
