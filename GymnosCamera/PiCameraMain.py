# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import Predictors

class PiCameraMain:
    def __init__(self):
        # initialize the HOG descriptor/person detector
        self.IM_WIDTH = 640
        self.IM_HEIGHT = 480
        self.camera = PiCamera()
        self.rawCapture = PiRGBArray(self.camera, size=(self.IM_WIDTH, self.IM_HEIGHT))
        self.camera.resolution = (self.IM_WIDTH, self.IM_HEIGHT)
        self.camera.framerate = 32

        # allow the camera to warm up
        time.sleep(0.1)

        # initialize the Predictors
        self.predictor = Predictors.Predictors()



    def run_loop(self):
        """
        This main loop will grab frames the camera and print it onto the screen
        """
        # capture frames from the camera
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # Retrieve frame as numpy array
            image = frame.array

            # Pass frame into the model
            image = self.predictor.hog_detector(image)

            # show the output images
            cv2.imshow("Video Feed", image)

            # clear the stream in preparation for the next frame
            self.rawCapture.truncate(0)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
