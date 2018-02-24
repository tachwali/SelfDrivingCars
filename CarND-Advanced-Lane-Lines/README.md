## Advanced Lane Lines Detection
Udacity Self Driving Car Nanodegree - Advanced Lane Lines Detection

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./output_images/calibration_image.jpg "Calibration"
[image2]: ./output_images/Undistorted_Image.jpg "Undistorted"
[image3]: ./output_images/Detected_Image.jpg "Road Transformed"
[image4]: ./output_images/Transformed_Image.jpg "Binary Example"
[image5]: ./output_images/Lane_Detection_Image.jpg "Warp Example"
[image6]: ./output_images/Filled_Image.jpg "Fit Visual"
[image7]: ./output_images/Edited_Image.jpg "Output"
[image8]: ./examples/warped_straight_lines.jpg "Warped Reference"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Camera calibration section of the IPython notebook located in "./Advanced_Find_Lanes.ipynb" and camera calibration functions in camera.py module.  

The calibration is performed on a list of calibration images. Each image shows a chessboard at a different orientation. The calibration process is comprised of two stages:

1) Convert the image to gray: To convert the image from 3D to 2D. The calibration process is the same in all image channels.

2) Find chessboard corners: By finding the (x,y) coordinates of the chessboard corners. This process is performed on all calibration images to collect a list of chesscorners at different orientations.

3) Camera clibration: which is the process that takes all corners found in step 2) and the the ideal corner locations and return the calibration matrix that can be used to correct (undistort) images taken by the same camera used to take the calibration images

The code below iterates over all calibration images and then undistort them using the calibration matrix "mtx" found by cv2.calibrateCamera function.
obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Those thresholding operations are done in detectors functions that can be found in `pipeline_functions.py` module. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_image()`, which can be found in `pipeline_functions.py` module. This function takes as inputs an image (`img`), and assume source (`src`) and destination (`dst`) points based on warped_straight_lines.jpg example image in the "example" folder shown below:

![alt text][image8]

I chose the hardcode the source and destination points in the following manner:

```python
src = np.array([[205, 720], [1120, 720], [745, 480], [550, 480]], np.float32)
dst = np.array([[205, 720], [1120, 720], [1120, 0], [205, 0]], np.float32)
```

I verified that my perspective transform was working by applying the `transform_image()` function on the detection binary image displayed earlier and obtained the following warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then the lines of the road are detected using histogram of a portion if the image. The peaks at the left and right of the histogram resembles the location of lines. A sliding window method is used to determine the lines location at different locations along the y axes of the image. The nonzero pixles around the line location are kept as a result of this method. The implementation of this method can be found in function `get_left_right_lane_points` in `pipeline_functions.py` module. The result of this detection algorithm can be shown in the image below:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature radius calculation is estimated based on the line points detected by the line detection algorithm described earlier. A second order polynomial coefficients are estimated to fit those points based on least mean square error criteion. A scaling factor is applied to idences of the line points to translate them into metric scale. The equation and scaling factor used to calculate the radius of left and right lines can be found in function `calculate_curvature` in `pipeline_functions.py` module. In addition, the distance from road center is also calculated in that function. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step using function `fill_area` in `viz.py` module. Here is an example of my result on a test image:

![alt text][image6]

and the estimated measurements are also displayed then on the image as shown below:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some of the challenges that I have faced in this project:
1- RGB and BGR color transformation when dealing with images read by matplotlib and cv2 library
2- Handling noise measurements resulted from inaccurate curvature measurements. I have decideded to use a Lane class to keep track of the measurements by storing a history of previous calculations and then use median operation to reject outliers. Also, to smooth the radius calculation further, I am taking the average of right and left curvature radius.
3- Deciding on the appropriate combination of gradient detectors was through trial and error. I have found that direction detector was redundant and kept only xy detector in combination with color and apply masking for black pixels. This combination appears to yield robust result. Direction detector can be limiting when dealing with sharp turns.
4- Future imrpovement may include using sanity check on the detected line points to calculate the curvature only upon their pass of that check. This check is partially implemented in Line class.


