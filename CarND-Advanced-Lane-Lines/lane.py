####################################
## classes for lane detections
####################################
import pipeline_functions as pf
import camera
import numpy as np
import viz


class Fifo():
    '''
    This is a simple fifo class to store
    lane measrurement over a time window
    of multiple images and calculates
    the median of those measurements to
    isolate measurement outliers
    '''
    def __init__(self, size):
        self.fifo = np.zeros(size)
        self.index = 0
    def push(self, x):
        fifo_size = len(self.fifo)
        self.fifo[self.index%fifo_size] = x
        self.index += 1
    def median(self):
        return np.median(self.fifo)
    def reset(self, size):
        self.fifo = np.zeros(size)
        

class Lane():
    '''
    This is a lane object that can be used to
    detect left and right lines in a road lane
    and to track those detections across consequent
    road images
    '''
    def __init__(self, mtx, dist, median_window):
        '''
        Lane object constructor
        Inputs
        mtx: cameraMatrix output from OpenCV `calibrateCamera` function
        dist: distortion Coefficients output from OpenCV `calibrateCamera` function
        median_window: median window size along which a smoothed measurement is calculated 
        ---
        Returns
        None
        '''
        self.poly_fit = None
        self.mtx = mtx
        self.dist = dist
        self.measurement_fifo_left = Fifo(history)
        self.measurement_fifo_right = Fifo(history)
        self.measurement_fifo_center = Fifo(history)

    def reset(self):
        '''
        Resets Lane object to initial state
        '''
        self.measurement_fifo_left.reset(history)
        self.measurement_fifo_right.reset(history)
        self.measurement_fifo_center.reset(history)
                
    def lane_detector(self, img):        
        '''
        Apply left and right line detection on input image and calculate the radius of
        their curvature and returns those measeurements edited on the output image
        Inputs
        img: input road image as np.array
        ---
        Returns
        road image as np.array with highlighted detected lane on text of lane measurements
        '''
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
        self.update_measurements(measurements)
        smoothed_measurements = self.smoothed_measurements()
        filled = viz.fill_area(img, self.poly_fit, inverse_matrix)

        edited = viz.write_info(filled, smoothed_measurements['curve'],smoothed_measurements['center'])
        return edited

    def update_measurements(self, measurements):
        '''
        Store new measurements
        Inputs
        measurements: this is the output of calculate_curvature function in pipeline_functions module
                      which is three measurements in a dictionary format
                      'left' : the radius of the left line.
                      'right' : the radius of the right line.
                      'center' : the offset of the vehicle from the center of the lane.
        ---
        Returns
        None
        '''
        self.measurement_fifo_left.push(measurements['left'])
        self.measurement_fifo_right.push(measurements['right'])
        self.measurement_fifo_center.push(measurements['center'])
            
    def smoothed_measurements(self):
        '''
        Obtain average radius of the lane curvature and offset from center
        ---
        Returns
        Dictionary of 'curve' which the average of left and right line radius and 'center'
        which is the offset from lane center
        '''
        left = self.measurement_fifo_left.median()
        right = self.measurement_fifo_left.median()
        curve = (left+right)/2.0
        center = self.measurement_fifo_center.median()        
        return {"curve":curve, "center":center}
               

    def direction_sanity_check(self, left, right):
        """
        FIXEME: UNFINISHED FUNCTION
        Checking that they have 
        Checking that they are separated by approximately the right distance horizontally
        Checking that they are roughly parallel
        """
        ## Check 1: similar curvature 
        check1 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff > 0.3:
            check1 = False
    
        ## Check 2: correct spacing
        check2 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff < 0.3:
            check2 = False

        ## Check 3: roughly parallel
        check3 = True
        rad_left = left.radius_of_curvature
        rad_right = right.radius_of_curvature
        rad_max = np.max(rad_left, rad_right)
        relative_diff = np.abs(rad_left - rad_right)/rad_max
        if relative_diff < 0.3:
            check3 = False
    
        return check1 and check2 and check3


