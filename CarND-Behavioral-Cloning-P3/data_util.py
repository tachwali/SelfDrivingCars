import cv2
import pandas as pd
import numpy as np
import csv
import random
import platform
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


project_path =''

## reading input data paths in from the csv

def get_train_validate_lines(csv_file_path):
    ''' reading all lines of input csv file data, shuffle lines and split them into training and validation lines  
        csv_file_path: path to csv file
    Returns
        train_lines: training lines as numpy array 
        validation_lines: validation lines as numpy array
    '''

    lines = []

    ## read in the csv file with images and steering data
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        # skip first line
        first_line = next(reader)
        for line in reader:
            lines.append(line)

    ## shuffle the observations
    lines_shuffled = shuffle(lines)

    ## split into training and validation data sets
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size=0.2)

    return np.array(train_lines), np.array(validation_lines)


## convert color of the input image to YUV as mentioned in nvidia paper and crop image
def preprocess_image(img, color_conversion=cv2.COLOR_BGR2YUV):
    ''' convert color of the input image and crop
    Arguments
        img: individual image array
        color_conversion: color conversion to be performed

    Returns
        cropped_img: cropped and color corrected image array
    '''

    ## convert the color of the image 
    converted_img = cv2.cvtColor(img,color_conversion)

    ## crop the image 
    cropped_img = converted_img[60:140,:,:]

    return cropped_img

## data generator
def generate_data(observations, batch_size=128):
    ''' data generator in batches to be fed into the Keras fit_generator object
    Arguments
        observations: the array of observation data that is to be split into batches and read into image arrays
        batch_size: batches of images to be fed to Keras model

    Returns
        X: image array in batches as a list
        y: steering angle as a list 
    '''

    ## applying correction to left and right steering angles
    steering_correction = 0.2 
    
    ## set up generator
    while True:
        for offset in range(0, len(observations), batch_size):
            batch_obs = shuffle(observations[offset:offset+batch_size])

            center_images = []
            left_images = []
            right_images = []

            steering_angle_center = []
            steering_angle_left = []
            steering_angle_right = []

            ## loop through lines and append images path/ steering data to new lists
            for observation in batch_obs:
                center_image_path = './data/IMG/'+observation[0].split('/')[-1]
                left_image_path = './data/IMG/'+observation[1].split('/')[-1]
                right_image_path = './data/IMG/'+observation[2].split('/')[-1]

                center_images.append(preprocess_image(cv2.imread(center_image_path)))
                steering_angle_center.append(float(observation[3]))

                left_images.append(preprocess_image(cv2.imread(left_image_path)))
                right_images.append(preprocess_image(cv2.imread(right_image_path)))

                ## append the steering angles and correct for left/right images
                steering_angle_left.append(float(observation[3]) + steering_correction)
                steering_angle_right.append(float(observation[3]) - steering_correction)
	    
            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)

