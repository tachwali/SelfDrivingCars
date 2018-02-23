####################################
## classes for lane detections
####################################

import pipeline_functions as pf
import camera
import numpy as np

class Fifo():
    def __init__(self, size):
        self.fifo = np.zeros(size)
        self.index = 0
    def push(self, x):
        fifo_size = len(self.fifo)
        self.fifo[self.index%fifo_size] = x
        self.index += 1
    def median(self):
        return np.median(self.fifo)
    

class Line():
    def __init__(self, mtx, dist):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.poly_fit = None
        self.mtx = mtx
        self.dist = dist
        self.measurement_fifo_left = Fifo(15)
        self.measurement_fifo_right = Fifo(15)
        self.measurement_fifo_center = Fifo(15)
		
    def lane_detector(self, img):
		
        undistorted = camera.undistort_img(img, self.mtx, self.dist)
        detected = pf.detector_pipeline(undistorted)
        transformed, _, inverse_matrix = pf.transform_image(undistorted)
        transformed_binary = pf.detector_pipeline(transformed)
        if self.detected == False:
            points, _ = pf.get_left_right_lane_points(transformed_binary)            
            self.detected = True
        else:
            points = pf.get_next_left_right_lane_points(transformed_binary, self.poly_fit)
        self.poly_fit = pf.polyfit(points)
        measurements = pf.calculate_curvature(np.shape(img), points)
        median_measurements = self.update_measurements(measurements)
        filled = pf.fill_area(img, detected, self.poly_fit, inverse_matrix)

        edited = pf.write_info(filled, median_measurements)
        return edited

    def update_measurements(self, measurements):
        self.measurement_fifo_left.push(measurements[0])
        self.measurement_fifo_right.push(measurements[1])
        self.measurement_fifo_center.push(measurements[2])
        return [self.measurement_fifo_left.median(),
                self.measurement_fifo_left.median(),
                self.measurement_fifo_left.median()]


    def direction_sanity_check(self, left, right):
        """

        Checking that they have 
        Checking that they are separated by approximately the right distance horizontally
        Checking that they are roughly parallel
        """

		# Check 1: similar curvature 
        check1 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff > 0.3:
            check1 = False
	
		# Check 2: correct spacing
        check2 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff < 0.3:
            check2 = False

        # Check 3: similar curvature 
        check3 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff < 0.3:
            check3 = False
	
        return check1 and check2 and check3


