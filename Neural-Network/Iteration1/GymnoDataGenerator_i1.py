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
    # Initialization
    def __init__(self, list_IDs, labels, data_location, batch_size=32, dim=(32,32,32), n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_location = data_location
        self.on_epoch_end()

    # Updates Indexes after each epoch
    def on_epoch_end(self):
        if self.shuffle == True:
            self.list_IDs = shuffle(self.list_IDs)
        
    # Generates batches of samples from accessing the drive. (n_samples, *dim, channels)
    def __data_generation(self, list_IDs_temp):
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

    # Returns the number of batches we will have per epoch
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Chooses which labels to retrieve from the drive
    def __getitem__(self, index):
        # Generate data
        X, y = self.__data_generation(self.list_IDs[index*self.batch_size:(index+1)*self.batch_size])

        return X, y
