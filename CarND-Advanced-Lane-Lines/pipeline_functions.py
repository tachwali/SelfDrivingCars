####################################
## functions for lane detections
####################################

import numpy as np
import cv2
import matplotlib.pyplot as plt


def scale_to_uint(img, base):
    '''
    scale to 8-bit (0 - 255) and quantize it so that the 
	maximum pixel intensity corresponds	to maximum value 
	in the quantization base.
	Inputs
	img: 2D Numpy array of an image
	base: quantization base such as np.uint8 or np.uint16 
	---
	Returns 
 	Numpy array of scaled image
    '''
    abs_img = np.abs(img)
    max_abs_img = np.max(abs_img)
    max_uint = np.iinfo(base).max
    scaled = base(max_uint*img/max_abs_img) 
    return scaled 


def apply_threshold(img, min_threshold, max_threshold):
    '''
	Apply thresholding and returns a binary image where the 
	non-zero pixels of that binary image corresponds to pixel
	in the original image with intensity value between min and
	max threshold parameters
	Inputs
	img: 2D Numpy array of an image
	min_threshold: the minimum value of thresholding range 
	max_threshold: the maximum value of thresholding range
	---
	Returns
 	A 2D numpy array of a binary image with non zero pixels 
	at locations that fall within the min and max thresholds
    '''
    binary_output = np.zeros_like(img)
    binary_output[(img >= min_threshold) & (img <= max_threshold)] = 1
    return binary_output


def scale_and_apply_threshold(img, min_threshold, max_threshold):
    '''
    Quantize, scale input image then apply thresholding.
	The quantization base is uint8
	Inputs
	img: 2D Numpy array of an image
	min_threshold: the minimum value of thresholding range 
	max_threshold: the maximum value of thresholding range
	---
	Returns
 	A 2D numpy array of a binary image with non zero pixels 
	at locations that fall within the min and max thresholds
    '''
    abs_img = np.abs(img)

    # scale to 8-bit (0 - 255) then convert to type = np.uint8  
    scaled = scale_to_uint(abs_img, np.uint8)
    
    binary_output = apply_threshold(scaled, min_threshold, max_threshold) 
    return binary_output


def detector_xy(img, orient='x', thresh_min=0, thresh_max=255):
    '''
	Gradient detector along x or y direction. 
    Inputs
    img: input image in RGB format
    orient: set it to 'x' or 'y' to detect edges along horizontal and 
		    vertical direction respectively
    thresh_min: minimum absolute value of a gradient to detect an edge 
    thresh_max: maximum absolute value of a gradient to detect an edge 
    ---
    Returns
    A 2D numpy array of a binary image with non zero pixels 
	at locations with gradient that fall within the min and max thresholds
    '''
    ## convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    ## take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    binary_output = scale_and_apply_threshold(sobel, thresh_min, thresh_max)
      
    return binary_output


def detector_mag(img, sobel_kernel=3, thresh_min=0, thresh_max=255):   
    '''
	Gradient detector that combines detection along x and y direction. 
    Inputs
    img: input image in RGB format
    orient: set it to 'x' or 'y' to detect edges along horizontal and 
			vertical direction respectively
	sobel_kernel: size of sobel kernel, it must be an odd positive integer
    thresh_min: minimum absolute value of a gradient to detect an edge 
    thresh_max: maximum absolute value of a gradient to detect an edge 
    ---
    Returns
    A 2D numpy array of a binary image with non zero pixels 
	at locations with gradient that fall within the min and max thresholds
    ''' 
    ## convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # calculate the magnitude 
    sobelxy_mag = np.sqrt(sobelx**2 + sobely**2)
    
    binary_output = scale_and_apply_threshold(sobelxy_mag, thresh_min, thresh_max)
    
    return binary_output


def detector_direction(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):    
    '''
    Inputs
    img: input image in RGB format
	sobel_kernel: size of sobel kernel, it must be an odd positive integer
    thresh_min: minimum detected direction angle in radian    
    thresh_max: maximum detected direction angle in radian  
    ---
    Returns
	A 2D numpy array of a binary image with non zero pixels with gradient
	at direction that fall within the min and max thresholds
    '''
    ## convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    
    # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    
    # create a binary mask where direction thresholds are met
    binary_output = apply_threshold(dir_grad, thresh_min, thresh_max)    
    
    return binary_output


def detector_combined(img, sobel_thresh=(20,100), sobel_kernel=3, mag_thresh=(30,100)):
    '''
	This detector combines the detections of three gradient detectors and 
	mask the detection result to keep only the detection at non-black pixels	
    Inputs
    img: input image in RGB format
    sobel_thresh:
    sobel_kernel:
    mag_thresh:
    orient: set it to 'x' or 'y' to detect edges along horizontal and vertical direction respectively
    thresh_min: minimum absolute value of a gradient to detect an edge 
    thresh_max: maximum absolute value of a gradient to detect an edge 
    ---
    Returns
    A 2D numpy array of a binary image with non zero pixels with 
    at detection pixel locations.
    '''    
    binary_output_sobel = detector_xy(img, 'x', sobel_thresh[0], sobel_thresh[1])
    binary_output_mag = detector_mag(img, sobel_kernel, mag_thresh[0], mag_thresh[1])
    
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    black_color = cv2.inRange(hls_image, np.uint8([0,0,0]), np.uint8([75,255,255]))
    
    combined = np.zeros_like(binary_output_mag)
    combined[(((binary_output_sobel == 1) | (binary_output_mag == 1)) & (black_color != 1))] = 1
    
    return combined


def detector_color(img, color_thresh_input=(170, 255)):
    '''
	Color selection for yellow and white, using the HLS and HSV color space
    Inputs
    img: input image in RGB format
    thresh_min: minimum absolute value of Hue value to detect 
    thresh_max: maximum absolute value of Hue value to detect 
    ---
    Returns
    A 2D numpy array of a binary image with non zero pixels with 
    at detection pixel locations.
    '''
    ## convert to HLS color space and separate the S channel
    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls_image[:,:,1]
    s_channel = hls_image[:,:,2]
    
    ## convert to the HSV color space    
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## selecting colors yellow and white
    white_color = cv2.inRange(hls_image, np.uint8([10,200,0]), np.uint8([255,255,255]))
    yellow_color = cv2.inRange(hsv_image, np.uint8([15,60,130]), np.uint8([150,255,255]))
    
    ## combine yellow and white masks
    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    
    ## threshold color channel
    s_binary = apply_threshold(combined_color_images, color_thresh_input[0], color_thresh_input[1])
    
    l_binary = apply_threshold(l_channel, color_thresh_input[0], color_thresh_input[1])
    
    ## combined binaries
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((s_binary > 0) & (l_binary > 0)) | (combined_color_images > 0)] = 1

    return combined_binary


def detector_pipeline(img, sobel_thresh=(40,100), sobel_kernel=3, mag_thresh=(30,100), color_thresh_input=(170, 255)):
    '''
    This detector pipeline combines the detection from color detector and combined edge detectors  
    Inputs
    img: input image in RGB format
    sobel_thresh: a tuple of two intergers from 0 to 255 for min and max detection threshold based on sobel kernel
    sobel_kernel: size of sobel kernel, this should be an odd integer value
    mag_thresh: a tuple of two intergers from 0 to 255 for min and max detection threshold based on sobel kernel
    color_thresh_input: min and max detection threshold of Hue values
    ---
    Returns
    A 2D numpy array of a binary image with non zero pixels with 
    at detection pixel locations.
    '''    
    ## denoise image
    img = cv2.fastNlMeansDenoisingColored(img,7,13,21,5)
    
    ## sobel, magnitude, direction threshold binary
    combined_thresh_binary = detector_combined(img, sobel_thresh=(40,100), sobel_kernel=3, mag_thresh=(30,100))
    
    ## color threshold binary
    color_thresh_binary = detector_color(img, color_thresh_input=color_thresh_input)
    
    ## combine the binaries
    combined_binary = np.zeros_like(color_thresh_binary)
    combined_binary[(color_thresh_binary == 1) | (combined_thresh_binary == 1)] = 1
    
    return combined_binary
    

def transform_image(img):
	'''
	It would be nice to have calibration chess board to get the prospective automaticallly from detected corner points.
	However, we found this calibration points through manual calibration found in notebook ????

	'''
    ## define image shape
	img_size = (img.shape[1], img.shape[0])

	## define source and destination points
	src = np.array([[205, 720], [1120, 720], [745, 480], [550, 480]], np.float32)
	dst = np.array([[205, 720], [1120, 720], [1120, 0], [205, 0]], np.float32)

	## perform transformation
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	return warped, M, Minv


def histogram(img):
    '''
    Take a histogram of the bottom half of the image
    '''
	return np.sum(img[img.shape[0]//2:,:], axis=0)


def get_left_right_lane_points(img):
	# img is a warped binary image
	img_width = img.shape[1]
	img_height = img.shape[0]
	# Take a histogram of the bottom half of the image
	hist = histogram(img)

	# Create an output image to draw on and visualize the result
	out_img = np.uint8(np.dstack((img, img, img))*255)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines

	midpoint = np.int(img_width/2)

	leftx_base = np.argmax(hist[:midpoint])
	rightx_base = np.argmax(hist[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(img_height/nwindows)

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image
		cv2.rectangle(out_img,
					  pt1 = (win_xleft_low,win_y_low),
                      pt2 = (win_xleft_high,win_y_high),
		              color = (0,255,0), 
                      thickness = 2) 
		cv2.rectangle(out_img,
                      pt1 = (win_xright_low,win_y_low),
                      pt2 = (win_xright_high,win_y_high),
		              color = (0,255,0), 
                      thickness = 2) 
		
        # Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		
        # Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	left_points = {'x': leftx,'y': lefty}
	right_points = {'x': rightx, 'y': righty } 
	return {'left':left_points, 'right':right_points}, out_img

def polyfit(points):
	#TODO: merge it with get_left_right_lane_points
	# Fit a second order polynomial to each
	left_fit = np.polyfit(points['left']['y'], points['left']['x'], 2)
	right_fit = np.polyfit(points['right']['y'], points['right']['x'], 2)
	return {'left':left_fit, 'right':right_fit}	


def visualize_poly(handle, out_img, poly_fit, points):
	leftx = points['left']['x']
	lefty = points['left']['y']
	rightx = points['right']['x']
	righty = points['right']['y']
	
	left_fit = poly_fit['left']
	right_fit = poly_fit['right']

	#ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]
	handle.imshow(out_img)
	handle.plot(left_fitx, ploty, color='yellow')
	handle.plot(right_fitx, ploty, color='yellow')

def get_next_left_right_lane_points(img, poly_fit):
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	left_fit = 	poly_fit['left']
	right_fit = poly_fit['right']

	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_points = {'x': leftx,'y': lefty}
	right_points = {'x': rightx, 'y': righty } 
	
	# Generate x and y values for plotting
	# ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	return {'left':left_points, 'right':right_points}


def calculate_curvature(img_shape, points):
	"""
	Returns the curvature of the polynomial `fit` on the y range `yRange`.
	"""
	height = img_shape[0]
	width = img_shape[1]
    
	## max y
	y_eval = height - 1
	
    # extract left and right line pixel positions
	leftx = points['left']['x']
	lefty = points['left']['y']
	rightx = points['right']['x']
	righty = points['right']['y']
	
	# It is assumed that a projected lane is 30 meters long and 3.7 meters wide
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	
	## distance from center
	center_idx = width//2
	identified_lanes_center_idx = (min(leftx) + max(rightx))//2

	dist_from_cent = np.abs(center_idx - identified_lanes_center_idx)*xm_per_pix
	return np.array([left_radius, right_radius, dist_from_cent])

## draw lane area
def fill_area(img, bin_img, poly_fit, inverse_matrix):
	left_fit_poly = poly_fit['left']
	right_fit_poly = poly_fit['right']
	## create copy of image
	img_copy = np.copy(cv2.resize(img, (1280, 720)))

	## define range
	ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(bin_img).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	## fit lines
	left_fitx = left_fit_poly[0]*ploty**2 + left_fit_poly[1]*ploty + left_fit_poly[2]
	right_fitx = right_fit_poly[0]*ploty**2 + right_fit_poly[1]*ploty + right_fit_poly[2]

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, inverse_matrix, (img_copy.shape[1], img_copy.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img_copy, 1, newwarp, 0.3, 0)

	return result

## write info about curvature
def write_info(area_img, curvature_output):

	## copy area image
	area_img_copy = np.copy(area_img)

	## format text for overlay
	left_text = "Left Curve Radius: {0:.2f}m".format(curvature_output[0])
	right_text = "Right Curve Radius: {0:.2f}m".format(curvature_output[1])
	dist_text = "Distance from Center: {0:.2f}m".format(curvature_output[2])

	## area_img writing
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(area_img_copy, left_text, (60,90), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
	cv2.putText(area_img_copy, right_text, (60,140), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
	cv2.putText(area_img_copy, dist_text, (60,190), font, 1.25, (255,255,255), 2, cv2.LINE_AA)

	return area_img_copy




