# Vehicle Detection

The goal of this project is to write a software pipeline to detect vehicles in a video (see results in final_output_project_video.mp4).  


Project Resources
---
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples used to train the classifier. 

Example images in the `test_images` folder are used for testing each stage of the vehicle detection pipeline. The output of each stage using those testing images are stored in the folder called `ouput_images`.  

The project code is comprised of:
* Vehicle_Detection notebook which demonstrates the vehicle detection pipeline which utilizes the functions in the following python modules
* features.py: contains the essential feature extraction functions such as color, spatial and HOG features
* viz.py: contains visualization helper functions
* utils.py: contains some other utility functions to perform color conversion and data analysis.
* search.py: contains the windowing functions
* detection.py: contains the vehicle detection class

The Project
---

The steps of this project are the following:

* Step1: Reading training images and obtain basic information about those images such as number of samples and image sizes.
* Step2: Feature Extraction:  Perform a Histogram of Oriented Gradients (HOG) feature extraction as well as colot and spatial bin information of each image
* Step3: Classifier selection and training
* Step4: Develop window search algorithm 
 use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



---

[//]: # (Image References)
[image1]: ./output_images/car_image.png
[image2]: ./output_images/car_HOG_image.jpg
[image2]: ./output_images/notcar_image.jpg
[image2]: ./output_images/notcar_HOGimage.jpg
[image3]: ./output_images/sliding_window_0.jpg 
[image4]: ./output_images/sliding_window_1.jpg
[image4]: ./output_images/sliding_window_2.jpg
[image4]: ./output_images/sliding_window_3.jpg
[image4]: ./output_images/sliding_window_4.jpg
[image4]: ./output_images/sliding_window_5.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./output_images/final_output_project_video.mp4


---


## Step 1: 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


1- set HOG 1-2 cell sizes similar to car features : e.g. 1 x cell == tail light. 2 x cells == registration plate.
2- set HOG block size should be roughly able to capture the larger car features e.g. back window, door, bumper, etc.
3- Warning: when dealing with image data that was extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. For the subset of images used in the next several quizzes, this is not a problem, but to optimize your classifier for the project, you may need to worry about time-series of images!
4-If you have to append training data then make sure to cut it at 64x64 since search windows asssume that

5-increasing the overlap will increase the multiple detection over the same car
6- the size of the window was also important in obtaining good detection

