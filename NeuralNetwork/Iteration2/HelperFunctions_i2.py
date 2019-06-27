import csv
import math  # for mathematical operations
import os

import matplotlib.pyplot as plt  # for plotting the images
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def standardize_file_names(source):
    """
    Modifys a directorys files to be named in an incremental fashion
    Ex) source: video1, video2, video3
    Parameters
    ---------
    source: Source Folder of videos to process
    """
    # Convert File Names into a List
    video_list = os.listdir(source)
    print(video_list)

    # Switch into the Directory and rename all files
    os.chdir(source)
    for i in range(len(video_list)):
        os.rename(video_list[i], 'video' + str(i) + '.MOV')


def init_labels_csv(labels_location):
    """
    Initializes the labels csv file in the path given.
    Parameters
    ---------
    labels_location: Location to create csv file
    ex) /content/drive/My Drive/GYMNOS/Video Dataset/labels.csv
    """
    # Initialize the Csv we will store our labels
    with open(labels_location, mode='w') as csv_file:
        fieldnames = ['Video_ID', 'Class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def append_to_labels_csv(labels_location, data_path, class_number):
    """
    Appends data points to the Labels CSV
    Ex) {'/path/file1': 0, '/path/file2': 0, '/path/file3': 1, '/path/file4': 1}
    Parameters
    ---------
    labels_location: Location of the csv labels cvs
    data_path: Path to data you want to label
    class_number: Label you would like to give each data point in the Path
    """
    # Initialize the Csv we will store our labels
    videos = []
    for vid in os.listdir(data_path):
        videos.append(data_path + vid)

    with open(labels_location, mode='a') as csv_file:
        fieldnames = ['Video_ID', 'Class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for video in videos:
            writer.writerow({'Video_ID': video, 'Class': class_number})


def generate_partitions_csv(partitions_location, labels_csv):
    """
    Generates a Partitions CSV
    Ex) {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    Parameters
    ---------
    partitions_location: Location to store the generated partitions csv
    labels_csv: Location of Labels CSV which we will use to generate partitions
    """

    # Initialize dictionaries for storting metadata of the dataset
    train = []
    validation = []
    test = []

    # Read in the Labels csv
    data = pd.read_csv(labels_csv)

    # Shuffle the data and
    data = shuffle(data)

    # Break the data into Partitions for train, validation and test
    dataset_size = len(data)
    training_size = math.floor(dataset_size * 0.70)  # 70% of videos are for training
    validation_size = math.floor(dataset_size * 0.20)  # 20% are for validation

    count = 0  # Counts the current frame we are on
    for index, video in data.iterrows():
        if count < training_size:
            train.append(video.Video_ID)
        elif count < (training_size + validation_size):
            validation.append(video.Video_ID)
        else:  # 10% are for testing
            test.append(video.Video_ID)

        count += 1

    # Store the sets into the dictionary
    print(len(train))
    print(len(validation))
    print(len(test))

    # Save the partitions dictionary in a csv
    with open(partitions_location, 'w') as csvfile:
        fieldnames = ['Partition', 'Dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Partition': 'train', 'Dataset': train})
        writer.writerow({'Partition': 'validation', 'Dataset': validation})
        writer.writerow({'Partition': 'test', 'Dataset': test})


def read_labels_csv(source):
    """
    Returns a dictionary of Labels and their classes
    Ex) {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    Parameters
    ---------
    source: Location of the labels CSV
    Returns
    ---------
    labels: {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    """
    labels = {}
    # Load in the Labels from the CSV
    with open(source, mode='r') as labels_csv:
        csv_reader = csv.DictReader(labels_csv)
        for row in csv_reader:
            labels[row["Video_ID"]] = row["Class"]

    return labels


def read_partition_csv(source):
    """
    Returns a dictionary of partitions and the frames in each partition
    Ex) {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    Parameters
    ---------
    source: Location of the partitions CSV
    Returns
    ---------
    partition: {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    """
    partition = {}
    # Load in the Partitions from the CSV
    with open(source, mode='r') as partitions_csv:
        csv_reader = csv.DictReader(partitions_csv)
        for row in csv_reader:
            dataset_as_string = row["Dataset"]  # Returns Row as a String
            partition[row["Partition"]] = dataset_as_string[2:-2].split("', '")

    return partition


def show_images(images, cols=1, titles=None):
    """
    Display a list of images in a single figure with matplotlib.
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    cols (Default = 1): Number of columns in figure (number of rows is set to np.ceil(n_images/float(cols))).
    titles: List of titles corresponding to each image. Must have the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
