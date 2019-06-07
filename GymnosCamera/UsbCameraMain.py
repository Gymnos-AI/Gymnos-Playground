# Continuously capture frames and perform object detection on them
import cv2


class UsbCameraMain:
    def __init__(self):
        # initialize the camera
        self.camera = cv2.VideoCapture(0)

    def run_loop(self):
        """
        This main loop will grab frames the camera and print it onto the screen
        """
        # capture frames from the camera
        while True:
            ret, image = self.camera.read()

            # show the output images
            cv2.imshow("Video Feed", image)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
