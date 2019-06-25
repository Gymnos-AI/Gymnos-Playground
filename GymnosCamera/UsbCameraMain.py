# Continuously capture frames and perform object detection on them
import time
import cv2
import numpy as np
from GymnosCamera import Predictors

# TODO: Fetch this from a json file, currently being mocked with this dict
CAMERA_MACHINES = {
    "Main Floor": [
        [0.1, 0.1, 0.3, 0.8, "Squat-Rack"],
        [0.75, 0.2, 0.95, 0.8, "Bench"]
    ]
}

CURRENT_LOCATION = "Main Floor"
iou_threshold = 0.01
time_threshold = 2  # how many seconds until machine is sure you are in or out


class UsbCameraMain:
    def __init__(self):
        # initialize the camera
        self.camera = cv2.VideoCapture(0)
        self.camera_height = 256
        self.camera_width = 256
        self.time_used = {
            "Squat-Rack": 0,
            "Bench": 0
        }
        self.inside = {
            "Squat-Rack": False,
            "Bench": False
        }
        self.using = {
            "Squat-Rack": False,
            "Bench": False
        }
        self.time_changed = {
            "Squat-Rack": 0,
            "Bench": 0
        }
        self.time_elapsed = {
            "Squat-Rack": 0,
            "Bench": 0
        }

        # initialize the Predictors
        self.predictor = Predictors.Predictors('YOLOV3')

    def run_loop(self):
        """
        This main loop will grab frames the camera and print it onto the screen
        """
        while True:
            ret, image = self.camera.read()
            image = cv2.resize(image, (self.camera_height, self.camera_width))

            # run predictions every 10 frames to help keep up with live stream pace
            # if count % 10 == 0:
            list_of_coords = self.predictor.yolo_v3_detector(image)
            for (topX, leftY, bottomX, rightY) in list_of_coords:
                cv2.rectangle(image, (topX, leftY), (bottomX, rightY), (0, 0, 255), 2)

            list_of_stations = CAMERA_MACHINES[CURRENT_LOCATION]
            font = cv2.FONT_HERSHEY_SIMPLEX

            for (a, b, c, d, name) in list_of_stations:
                w, h = self.camera_width, self.camera_height
                topX, leftY, bottomX, rightY = int(a * w), int(b * h), int(c * w), int(d * h)

                # display machine areas
                cv2.rectangle(image, (topX, leftY), (bottomX, rightY), (0, 255, 0), 1)
                cv2.putText(image, name + " " + str(self.time_used[name]) + "s", (topX, int((leftY + rightY) / 2)),
                            font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # calculate iou for boxes
                for (HtopX, HleftY, HbottomX, HrightY) in list_of_coords:
                    if calculate_iou((topX, leftY, bottomX, rightY),
                                     (HtopX, HleftY, HbottomX, HrightY)) > iou_threshold:
                        # Count seconds somebody is in the machine
                        if self.using[name]:
                            cv2.rectangle(image, (topX, leftY), (bottomX, rightY), (0, 255, 0), 3)
                        if self.inside[name]:
                            diff = time.time() - self.time_changed[name]
                            if diff > time_threshold:
                                self.using[name] = True
                                self.time_elapsed[name] = self.time_changed[name]
                        else:
                            self.inside[name] = True
                            self.time_changed[name] = time.time()
                    else:
                        if self.inside[name]:
                            self.inside[name] = False
                            self.time_changed[name] = time.time()
                        else:
                            diff = time.time() - self.time_changed[name]
                            if diff > time_threshold and self.using[name]:
                                self.using[name] = False
                                self.time_used[name] += time.time() - self.time_elapsed[name] - time_threshold

            image = np.asarray(image)
            image = cv2.resize(image, (900, 900))
            cv2.imshow("Video Feed", image)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break


def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    # print(iou)
    return iou
