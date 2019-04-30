import pandas as pd
from sklearn.utils import shuffle
import csv
import cv2
import keras
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.utils.data_utils import Sequence
from random import randint


class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, frames_per_video, frame_dim, batch_size, n_classes, shuffle=True):
        """
          Initialization

          :param list_IDs:
          :param labels:
          :param frames_per_video:
          :param frame_dim:
          :param batch_size:
          :param n_classes:
          :param shuffle:
          """
        self.list_IDs = list_IDs
        self.labels = labels
        self.frames_per_video = frames_per_video
        self.frame_dim = frame_dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates Indexes after each epoch

        :return:
        """
        if self.shuffle == True:
            self.list_IDs = shuffle(self.list_IDs)

    def extract_frames_from_video(self, source):
        """
        Extracts frames from a video

        Parameters
        ---------
        source: The video we would like to retrieve frames from

        Returns
        ---------
        video_frames: Frames from the video that was passed in
        """
        video_frames = []

        cap = cv2.VideoCapture(source)  # capturing the video from the given path
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pos = randint(0, length - 140)
        cap.set(1, pos)

        frames_to_extract = self.frames_per_video
        frame_dim = self.frame_dim

        # Pull franes from videos until we have reached the amount we want to extract
        while cap.isOpened() and len(video_frames) < frames_to_extract:
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break

            # We are capturing at 28 frames per second.
            # If we want to capture every 0.2 seconds we will take every 5 frames
            if frame_id % 14 == 0:
                resized = cv2.resize(frame, frame_dim)  # Reads as BGR
                destRGB = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
                video_frames.append(np.rot90(destRGB, 3))  # Rotate the image to an upright position

        cap.release()
        return np.array(video_frames)

    def __data_generation(self, list_IDs_temp):
        """
        Generates batches of samples from accessing the drive. (n_samples, *dim, channels)

        :param list_IDs_temp:
        :return:
        """
        # Initialization
        X = np.empty((self.batch_size, *self.frame_dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.extract_frames_from_video(ID)

            # Store class
            y[i] = self.labels[ID]

        return preprocess_input(X, mode='tf'), keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        """
        Returns the number of batches we will have per epoch

        :return:
        """
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Chooses which labels to retrieve from the drive

        :param index:
        :return:
        """
        X, y = self.__data_generation(self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size])

        return X, y
