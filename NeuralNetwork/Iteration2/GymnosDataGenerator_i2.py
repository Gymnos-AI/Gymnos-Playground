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
    def __init__(self, list_IDs, labels, frames_per_video, frame_strides, frame_dim, batch_size, n_classes, shuffle=True):
        """
          Initialization

          :param list_IDs: List of all Videos the DataGenerator has access to
          :param labels: List of all Videos and their Labels as a dictionary
          :param frames_per_video: Amount of frames to extract per video
          :param frame_strides: Amount of frames in-between samples
          :param frame_dim: Width and Height of the frames
          :param batch_size: Amount of videos to pull frames from
          :param n_classes: Number of different labels in the Dataset
          :param shuffle: If true, we will shuffle the Dataset on every epoch
          """
        self.list_IDs = list_IDs
        self.labels = labels
        self.frames_per_video = frames_per_video
        self.frame_strides = frame_strides
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

    def __data_generation(self, list_IDs_temp):
        """
        Generates batches of samples from accessing the drive. (n_samples, *dim, channels)

        :param list_IDs_temp: Ex. list_IDs[0:32]

        :return X: A numpy array of the batch as (batch_size, frames_per_vid, *frame_dim, channels)
        :return y: One-hot encoding of the labels
        """
        # Initialization
        X = np.empty((self.batch_size, self.frames_per_video, *self.frame_dim, 3))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.extract_frames_from_video(ID)

            # Store class
            y[i] = self.labels[ID]

        return preprocess_input(X, mode='tf'), keras.utils.to_categorical(y, num_classes=self.n_classes)

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
        # If the video is longer than 8 seconds, start at a random position in the video
        if length > 180:
            pos = randint(0, ((length - 50) - (self.frame_strides * self.frames_per_video)))
            cap.set(1, pos)
        # Start at the beginning
        else:
            cap.set(1, 0)

        # Pull frames from videos until we have reached the amount we want to extract
        while cap.isOpened() and len(video_frames) < self.frames_per_video:
            frame_id = cap.get(1)  # Get current frame number
            ret, frame = cap.read()
            if ret != True:
                cap.set(1, 0)  # If we hit the end start grabbing from the start again

            # We are capturing at 28 frames per second.
            # If we want to capture every 0.2 seconds we will set frame_strides = 6
            elif frame_id % self.frame_strides == 0:
                resized = cv2.resize(frame, self.frame_dim)  # Reads as BGR
                dest_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
                video_frames.append(np.rot90(dest_rgb, 3))  # Rotate the image to an upright position

        cap.release()
        return np.array(video_frames)
