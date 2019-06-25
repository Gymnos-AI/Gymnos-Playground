import cv2                       # for capturing videos
import math                      # for mathematical operations
import matplotlib.pyplot as plt  # for plotting the images
import csv
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os


def extract_frames_from_directory(count, source, destination):
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
        video_file = source + video # Retrieve a video from the OverHeadPress
        cap = cv2.VideoCapture(video_file)   # capturing the video from the given path
        dim = (224, 224)

        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break

            # We are capturing at 28 frames per second. 
            # If we want to capture every 0.2 seconds we will take every 5 frames
            if frame_id % 8 == 0:
                filename ="frame%d.jpg" % count
                count+=1
                resized = cv2.resize(frame, dim)
                cv2.imwrite(destination + filename, resized)

        cap.release()
        print ("Finished processing: " + video + ". Ended at video: " + str(count))

        
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
        os.rename(video_list[i], 'video'+ str(i) + '.MOV')

        
def generate_labels_csv(csv_location, *args):
    """
    Generates a Labels CSV
    Ex) {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    
    Parameters
    ---------
    csv_location: Location to store the generated csv
    
    *args: Total size of each class. Order each class in the order you would like them to be labeled as.
    Ex) Squats (class 0) = 400, Bench (class 1) = 600
    
    generate_labels_csv(location, 400, 600)
    """
    os.chdir(csv_location) # Navigate into the right directory

    # Initilize and open the Labels csv
    with open('labels.csv', mode='w') as csv_file:
        fieldnames = ['Frame_ID', 'Class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        label = 0
        for classes in args:
            # Write into the CSV the frames with their associated class
            for i in range(count, classes):
                writer.writerow({'Frame_ID': 'frame' + str(i) + '.jpg', 'Class': label})

            # Increment label and count
            count = classes
            label += 1


def generate_partitions_csv(csv_location, labels_csv):
    """
    Generates a Partitions CSV
    Ex) {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    
    Parameters
    ---------
    csv_location: Location to store the generated csv
    
    labels_csv: Location of Labels CSV which we will use to generate partitions
    """
    os.chdir(csv_location) # Navigate into the right directory
    
    # Initialize dictionaries for storting metadata of the dataset
    count = 0         # Counts the current frame we are on
    train = []
    validation = []
    test = []

    # Read in the Labels csv
    data = pd.read_csv(labels_csv)

    # Shuffle the data and 
    data = shuffle(data)

    # Break the data into Partitions for train, validation and test
    dataset_size = len(data)
    training_size = math.floor(dataset_size * 0.70)    # 70% frames are for training
    validation_size = math.floor(dataset_size * 0.20)  # 20% are for validation

    for index, frames in data.iterrows():
        if count < training_size:  
            train.append(frames.Frame_ID)
        elif count < (training_size + validation_size):                 
            validation.append(frames.Frame_ID)
        else:                                 # 10% are for testing
            test.append(frames.Frame_ID)

        count += 1

    # Store the sets into the dictionary
    print(len(train))
    print(len(validation))
    print(len(test))

    # Save the partitions dictionary in a csv
    with open('partitions.csv', 'w') as csvfile:
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
            labels[row["Frame_ID"]] = row["Class"]
    
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
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
