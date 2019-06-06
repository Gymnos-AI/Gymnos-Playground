# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time


class PiCameraMain:
	def __init__(self):
		# initialize the HOG descriptor/person detector
		self.hog = cv2.HOGDescriptor()
		self.IM_WIDTH = 640
		self.IM_HEIGHT = 480
		self.camera = PiCamera()
		self.rawCapture = PiRGBArray(self.camera, size=(self.IM_WIDTH, self.IM_HEIGHT))

		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.camera.resolution = (self.IM_WIDTH, self.IM_HEIGHT)
		self.camera.framerate = 32

		# allow the camera to warm up
		time.sleep(0.1)

	def run_loop(self):
		"""
		This main loop will grab frames the camera and print it onto the screen
		"""
		# capture frames from the camera
		for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
			# Retrieve frame as numpy array
			image = frame.array

			# Pass frame into the model
			# image = self.person_detector(image)

			# show the output images
			cv2.imshow("Video Feed", image)

			# clear the stream in preparation for the next frame
			self.rawCapture.truncate(0)

			# Press 'q' to quit
			if cv2.waitKey(1) == ord('q'):
				break

	def person_detector(self, to_predict):
		"""
		Takes in a frame, runs it through a model and returns the results

		:param to_predict: The frame passed into the model
		:return: frame after it is processed
		"""
		(rects, weights) = self.hog.detectMultiScale(to_predict, winStride=(4, 4), padding=(8, 8), scale=1.05)
		for (x, y, w, h) in rects:
			cv2.rectangle(to_predict, (x, y), (x + w, y + h), (0, 0, 255), 2)

		return to_predict
