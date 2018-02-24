####################################
## functions for visualization
####################################
import matplotlib.pyplot as plt
import numpy as np
import cv2


def display_image(img, handle=None, label=""):
    '''
    Display image with title label
    Inputs
    img: input image as numpy array of two or three dimension.
         if it is of two dimension, it is shown as gray image
         otherwise it is shown as colored image
    handle: matplotlib subplot axes, if None, the function will create one
    label: image title as string array 
    ---
    Returns
    None
    '''
    if handle is None:
        f, ax = plt.subplots(1, 1, figsize=(15, 9))
        f.tight_layout()
        display_image(img, ax, label)
    else:
        ndim = len(np.shape(img))
        if ndim == 2:
            handle.imshow(img, cmap='gray')
            handle.set_title(label, fontsize=20)
        else:
            handle.imshow(img)
            handle.set_title(label, fontsize=20)


def disply_transform(before, after, label1, label2):
    '''
    Display two images to demonstrate a transfomation
    Inputs
    before: image before transformation placed to the left as numpy array.
    after: image after transformation placed to the right as numpy array.
    label1: before image title as string array 
    label2: after image title as string array 
    ---
    Returns
    None
    '''
    ## plot the images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    f.tight_layout()
  
    display_image(before, ax1, label1)
    display_image(after, ax2, label2)

 
def write_info(area_img, curve, center):
    '''
    Display lane measurements on image
    Inputs
    area_img: image to be used to display measurement on
    curve: radius of curve as floating point number
    center: offset from lane center as floating point number
    ---
    Returns
    numpy array of image with writen measurement text
    '''
    ## copy area image
    area_img_copy = np.copy(area_img)

    ## format text for overlay
    curve_text = "Curve Radius: {0:.2f}m".format(curve)
    dist_text = "Distance from Center: {0:.2f}m".format(center)

    ## area_img writing
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(area_img_copy, curve_text, (60,90), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(area_img_copy, dist_text, (60,140), font, 1.25, (255,255,255), 2, cv2.LINE_AA)

    return area_img_copy


def fill_area(img, poly_fit, inverse_matrix):
    '''
    Draw lane area
    Inputs
    img: input image as numpy array
    poly_fit: Dictionary of two keys 'Left' and 'right' each holds a 3 elements numpy 
              array that contains the coefficients of second order polynomial
              This input can be obtained from polyfit function in pipeline_functions module  
    inverse_matrix: 3 x 3 matrix of a perspective transform from bird to driver view
                    This input is obtained from transform_image function in pipeline_functions 
                    module  
    ---
    Returns
    None
    '''
    left_fit_poly = poly_fit['left']
    right_fit_poly = poly_fit['right']
    
    height = img.shape[0]
    width = img.shape[1]
    ## create copy of image
    img_copy = np.copy(cv2.resize(img, (1280, 720)))
        
    ## define range
    ploty = np.linspace(0, height-1, height)

    # Create an image to draw the lines on
    warp_zero = np.zeros((height, width)).astype(np.uint8)
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


def visualize_poly(handle, out_img, poly_fit, points):
    '''
    Shows image with fitted polynomial line and detected line points
    This function can be used to examine the result of 
    get_left_right_lane_points, get_next_left_right_lane_points and
    polyfit functions in pipeline_functions module 
    Inputs
    handle: matplotlib subplot axes 
    out_img: numpy 2D array image that shows the windows used
             to detect left and right image
             This input can be obtained from get_left_right_lane_points function
             in pipeline_functions module 
    poly_fit: Dictionary of two keys 'Left' and 'right' each holds a 3 elements numpy 
              array that contains the coefficients of second order polynomial
              This input can be obtained from polyfit function in pipeline_functions module  
    points: Dictionary : 'Left': Indeces of detected left lane, 
            'Right': Indeces of detected right lane, 
    ---
    Returns
    None
    '''
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
