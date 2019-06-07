# Continuously capture frames and perform object detection on them
import cv2

while True:
    IM_WIDTH = 640
    IM_HEIGHT = 480
    camera = cv2.VideoCapture(0)

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, image = camera.read()

    # show the output images
    cv2.imshow("Video Feed", image)


    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
