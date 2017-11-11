# **Finding Lane Lines on the Road** 

---

**Summary**
The goal of this project is to develop an algorithm to analyze the geometry of detected lines on the 
road and infer an optimal left and right line that bound the path of the vehicle. The slope of two lines
and the gap between them are determined by the analysis done of detected lines.

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
Step 1: Gray scale conversion : this stage allows the following stages to operate on a single channel (gray color) rather than three channels (RGB colors). This saves alot of computations especially when the task does not depend on color information but on the rate of color changes. 

Parameters: None


Step 2: Gaussian blurring: this stage allows for more stable line detection as it smooth out the small variations in colors that are not likely to be of interests for line detection
 not 
Parameters: window size: based on the examples provided, it seems a window of size 7 was able to remove the unwanted edges detected by the following stage 


Step 3: Edge detection: this is a critical stage which highlight the areas of images that represent possible segments of actual lines. 

Parameters: Threshold: we followed a ratio of 1:3 as recommended for canny edge detector. 


Step 4: Define area of interest: this stage determines the area of interest where we want to detect lines.
That area is defined by a polygon (trapezoid). The base of trapezoid is the bottom of the image, and the height is almost half of the image. The top size is 2/3 of the base length.
This area is determined manually by looking at multiple examples and realizing that detecting lines in the sky of from the sides of the road is not wanted.

Parameters: 4 points that represents the corners of  trapezoid


Step 5: Line detection: use hough transform to detect lines from the edges information provided in step 3 and masked by stage 4. 

Parameters:
hough transform grid resultion rho, theta (left at their default values)
threshold :very low values results in noiser detection
min_line_len : we want this to be large enough to reject small lines detected from cracks on the road and other road artifacts. but not too large to miss detecting the lines on the road
max_line_gap = 5  :very low values results in noiser detection




In order to draw a single line on the left and right lanes, I modified the draw_lines() function by adding a mechanism to draw two longs lines on the right and left of the vehicle. The slope of those
lines are the average from detected lines.
However, to make the algorithm stable :
1- slopes at zero or infinity are rejected
2- slope with absolute value less than 0.5 is rejected (represent almost horizontal lines)



### 2. Identify potential shortcomings with your current pipeline

Shortcomings of the current algorithm:
1- Not stabe when cars or objects appear in the area of interest
2- Not stable when alot of crack or objects appear on the road
3- The limitation of slopes to be larger than 0.5 may avoid the algorithm to detect lines on sharp turns


### 3. Suggest possible improvements to your pipeline

Improvements:
1- Doing masking at earlier stage and run the following edge and line detection on a portion of the image to save computation
2- Determine the edge thresholds based on histogram adaptively
3- apply better slope calculation using kalman filtering mechanism. I actually tried it by having a slope update following this equation:
slope = alpha*slope + (1-alpha) new_slope 
but did not result in good performance. I am pretty sure this should work but I ran out of time on this,

