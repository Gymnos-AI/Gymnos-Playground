### Data Collection:
* We collected Data from two different locations (UFC gym and Golds)
* We used an iPhone 8 front facing camera with an upright orientation
* We captured two different workouts (Overhead press vs. Squats) at one angle (front view)
* Each datapoint was RGB (224, 224, 3)

### Data ETL (Extract, Transfer, Load)
* I used a **Keras DataGenerator** as my ETL algorithm (GymnosDataGenerator.py) 
 * This can be loaded using *from tensorflow.python.keras.utils.data_utils import sequence*
* For Extracting and Transferring, I created a *extract_frames_from_directory()* function which goes into a specified folder and extracts frames from each video in that folder every 0.2 seconds. It then loads these frames into another folder where I store all of my frames for each video.
  * Downside to this is that you end up with a really big directory of frames you have to label and maintain (More steps to the process which we want to avoid)
* I then load all of this Data in using my DataGenerator which is pretty generalized at the moment
 * One concern is that I have to rotate every image here to have them all standing upright (would be nice to fix this)

### Data labeling and partitioning
* I used a csv to store all of my labels and partitions
* For labels I have created *generate_labels_csv()* but this is hardcoded. It assumes each class is in a contiguos block of frame numbers (Which might still work).
* For partitions I have created *generate_partitions_csv()* but this is also hardcoded to generate partitions on frames

### Model training
* This model predicts the workout using only one frame. Which means we only use spatial cues
* I also used Keras VGG16 strictly as a feature extractor (froze all of it's weights).
* Relu as my activation for intermediate layers
* Dropout rate was 0.5
* Sigmoid as my last activation because this is a binary classifier
``` 
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 7, 7, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              25691136  
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 2050      
=================================================================
Total params: 40,407,874
Trainable params: 25,693,186
Non-trainable params: 14,714,688
_________________________________________________________________
```
* Benchmarked the training on a CPU, GPU and TPU
 * CPU: 80 seconds per epoch
 * GPU: 20 seconds per epoch
 * TPU: Couldn't even run it on there.. 

### Evaluating the model
* Was able to get to a 100% accuracy but that's cap. Reason being is although it did not see any of the frames from the test set. It did see a frame from the same video which will help it alot.

### What I need to improve
* Change picture dimensions to 256x256 to optimize data for TPU
* Design a new approach for extracting frames from a directory. Specifically:
 * Do not extract frames from videos and store them in a seperate folder. It adds a useless step to the process and it uses up space for no reason
* The above step will probably force me to update how I generate labels and partitions. 
