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

* Step1: Reading training images 
* Step2: Feature Extraction  
* Step3: Classifier selection and training
* Step4: Develop window search algorithm 
* Step5: Develop video stream processing pipeline

Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



---

[//]: # (Image References)
[image1]: ./output_images/car_image.jpg
[image2]: ./output_images/car_HOG_image.jpg
[image3]: ./output_images/notcar_image.jpg
[image4]: ./output_images/notcar_HOGimage.jpg
[image5]: ./output_images/sliding_window_0.jpg
[image6]: ./output_images/sliding_window_1.jpg
[image7]: ./output_images/sliding_window_2.jpg
[image8]: ./output_images/sliding_window_3.jpg
[image9]: ./output_images/sliding_window_4.jpg
[image10]: ./output_images/sliding_window_5.jpg
[video1]: ./output_images/final_output_project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First, the training images are analyzed initially to check for :

* Balanced training data: There are equal number of images for vehicle and non-vehicle images.
* Size of training images: To understand the scale (dimension) of input data. Each training image is found to be 64x64 pixels.

Three features were extracted from each image: color, spatial and HOG features. The feature extraction code can be found in [features.py](./features.py). 

The color features are extracted after converting each image into another color space that is less sensitive to brightness variation. YCrCb, HLS, and HSV are among the color spaces that are decouple color information from brightness information. However, YCrCb was chosen due its faster computational performance. A histogram of 32 bins is extracted for each of the three channels in the image. This yields a color feature vector of 32x3 = 96 color features

The spatial information is also extracted by taking a downsampled and flattened version (32x32x3) of the image. This yields spatial features of size 3072

The hog features are the last set of features extracted from the image. The number of possible orientations used was 9 which is a typical number used between the range of 6 and 12. Also, 8 pixels per cells and 2 cell per blocks were used in the setting of HOG feature extraction. This results in 7×7×2×2×9 = 1764 HOG feature samples. Since this extraction is done for all three channels, then the total number of HOG features is 1764x3 = 5292. Examples of HOG information are shown below:


![alt text][image1]  ![alt text][image2]


![alt text][image3]  ![alt text][image4]


Note: It seems there is no direct way to use imwrite an image after applying hot color-map to appear in the same way as it is shown with imshow. To better see the HOG features extracted, refer to the project notebook. 

In summary, the size of feature vector per image is 96 + 3072 + 5292 = 8460. This size is believed to be very excessive and should be reduced to be much less than the input image size (4096 pixels). However, due to the lack of time, I continued with using this feature vector since it yields good classification performance. 


#### 2. Explain how you settled on your final choice of HOG parameters.

I have initally used number of orientation to be 12. However, that made the feature vector size really huge and resulted in uncessarily long training time. Two cell per block seems to be a typical initial setting while 8 pixel per cell was a number that divides 64 (image side length). 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have used GridSearchCV to search for the best classifier setting. I function returns SVM classifier with RBF kernel and C = 10 to be the classifier with best test performance. However, I found that it runs much slower than Linear SVM.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The window search code can be found in [search.py](./search.py). The window search is limited to the lower half of the image to avoid false alarm comings from the top half of the image and to speed processing and feature extractions. The window size is set to 96 and overlap of 0.5. Larger overlap ration caused misdetection at that window size. This setting result is shown below. Note that this overlap setting caused multiple detections which could be cleaned up later through heatmap and thresholding. Below is the detection result using this setting.

![alt text][image5]  ![alt text][image6]  ![alt text][image7]

![alt text][image8]  ![alt text][image9]  ![alt text][image10]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The implementation of car detector is found in [detection.py](./detection.py). The pipeline starts by extacting the lower half of the image and, with the exception of HOG features, all other features described earlier are extracted through a window size 64x64 that matches the input training image size. HOG features are extracted once for the entire image and we extract the related portion of those features for the window being analyzed. Then detection is applied using the SVM RBF classifier and a heatmap is build to track the number of multiple detection that could be applied to the same car due to overlapping window search. To obtain the binary detection image, instead of using direct thresholding, the label function is used from scipy.ndimage.measurements module, which allows to distinguish between different vehicle detections by yielding different label regions (blobs) that corresponds to each vehicle. Then a bounding box is extracted for each region seperately using draw_labeled_boxes function from [viz.py](./viz.py)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/final_output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As described in the pipeline implementation. The positions of positive detections in each frame of the video are obtained. Then, a heatmap is created  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The results that show the heatmap from testing images can be found in the notebook.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The following implementation issues were found during working on this project:

1- Setting HOG cell sizes should be similar to car features : e.g. 1 x cell == tail light. 2 x cells == registration plate.

2- Setting HOG block size should be roughly able to capture the larger car features e.g. back window, door, bumper, etc.

3- When dealing with image data that was extracted from video, the classifier is dealing with sequences of images where the target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. For the subset of images used in the next several quizzes, this is not a problem, but to optimize your classifier for the project, time-series of images should be avoided.

4- When trying to augment training data, I have to make sure to cut additional images at 64x64 since search windows and remaining training images are at that size.

5- Increasing the overlap will increase the multiple detection over the same car

6- The size of the window was very critical and sensitive parameter in obtaining good detection.
