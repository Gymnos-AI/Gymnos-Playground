import pandas as pd
from sklearn.utils import shuffle
import csv
import keras
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.utils.data_utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, data_location, batch_size=32, dim=(32,32,32), n_classes=2, shuffle=True):
        """
        Initialization

        :param list_IDs:
        :param labels:
        :param data_location:
        :param batch_size:
        :param dim:
        :param n_classes:
        :param shuffle:
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_location = data_location
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates Indexes after each epoch

        :return:
        """
        if self.shuffle == True:
            self.list_IDs = shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        """
        Generates batches of samples from accessing the drive. (n_samples, *dim, channels)

        :param list_IDs_temp:
        :return:
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.rot90(img_to_array(plt.imread(self.data_location + ID)), 3) # Rotate image 3 times to the left to have it standing up right
            
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
        X, y = self.__data_generation(self.list_IDs[index*self.batch_size:(index+1)*self.batch_size])

        return X, y

    def extract_frames_from_directory(self, count, source, destination):
        """
        Extracts Frames from a directory of videos and stores them in a specified destination directory

        Parameters
        ---------
        count: Last proccessed frame number
        ex) count = 20, then the first processed frame will be frame20.jpg and they will be incremented from there

        source: Source Folder of videos to process

        destination: Destination Folder where frames are stored in
        """
        all_videos = os.listdir(source)
        print(all_videos)

        for video in all_videos:
            video_file = source + video  # Retrieve a video from the OverHeadPress
            cap = cv2.VideoCapture(video_file)  # capturing the video from the given path
            dim = (256, 256)

            while (cap.isOpened()):
                frame_id = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break

                # We are capturing at 28 frames per second.
                # If we want to capture every 0.2 seconds we will take every 5 frames
                if (frame_id % 8 == 0):
                    filename = "frame%d.jpg" % count
                    count += 1
                    resized = cv2.resize(frame, dim)
                    cv2.imwrite(destination + filename, resized)

            cap.release()
            print("Finished processing: " + video + ". Ended at video: " + str(count))
