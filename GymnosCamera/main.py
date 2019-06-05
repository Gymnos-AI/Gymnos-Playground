# Import packages
import cv2

# Initialize USB webcam feed
camera = cv2.VideoCapture(0)
# Continuously capture frames and perform object detection on them
while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = camera.read()

    '''
    Here we need to pass the frame through the model and get the results
    '''

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
