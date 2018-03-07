####################################
## utility functions 
####################################
import cv2
import matplotlib.image as mpimg


def data_look(car_list, notcar_list):
    '''
    takes in car/non-car lists and returns a dictionary of dataset information
    Inputs
    car_list: list of car sample file locations
    notcar_list: list of non-car sample file locations
    ---
    Returns
    Dictionary of information about input lists: number of samples of cars and non-cars,
    size of image sample, type of data
    '''
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

def convert_color(img, conv='RGB2YCrCb'):
    """
    Call various cv2 image space conversions
    Inputs
    img: to convert from
    conv: space to convert to
    ---
    Returns
    extracted image space
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


