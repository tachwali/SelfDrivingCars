####################################
## functions for feature extraction
####################################
import numpy as np
import matplotlib.image as mpimg
import cv2 
from skimage.feature import hog


def bin_spatial(img, size=(32, 32)):
    """
    Performs spatial binning on the image,
    Inputs
    :img: input image to take the spatial and color information into the feature vector
    :size: down-sampling the image from 64x64 to say 32x32 pixel images
    ---
    Returns
    stacked feature vector of 3 color channels, having been down sampled to lower resolution
    determined by `size` input.
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    
    return np.hstack((color1, color2, color3))


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    This is a wrapper of skimag.feature.hog() to compute Histogram of Oriented Gradients (HOG) 
    for an image `img` and get HOG features vector and optionally an visualisation of result.
    Inputs
    img: to compute hog() on
    orient: # of HOG orientation bins
    pix_per_cell: size in pixels of a cell.        
    cell_per_block: # of cells in each block.        
    vis: return an image of the HOG
    feature_vec: features will be flattened to a vector, set to False if you want to 
                 extract HOG features for the entire image to simplify HOG feature sampling
    ---
    Returns
    :return: feature vector and optionally `hog_image` if `vis` is set to True
    """
    # Call with 2 outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Else call the 1 output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
    return features


def color_hist(img, nbins=32):  
    """
    Create a separate histogram for each color channel spread across the number of bins
    Inputs
    img: to split into color channel histograms
    nbins: number of bins to group data into
    ---
    Returns 
    Concatenated individual histogram feature vectors
    """
    # Compute the histogram of the color channesl separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):
    """
    This function is used to extract features from a single image for car Classifier.
    For a single image, extract color features. Optionally down-sample, color histogram, calc HOG.

    Note : Classifier will need to use the appended features in this order, else will not work ;
        spatial,
        histogram,
        HOG

    imgs: file list of images
    color_space: type of space to extract
    spatial_size: down-sample size
    hist_bins: # of groupings
    orient: # of HOG orientation bins
    pix_per_cell: size in pixels of a cell
    cell_per_block: # of cells in each block
    hog_channel: 'ALL' 3 channels or just the 1 '0 index' channel
    spatial_feat: flag to optionally calc or not
    hist_feat: flag to optionally calc or not
    hog_feat: flag to optionally calc or not
    ---
    Returns 
    list of these feature vectors
    """
    # 1) Define an empty list to receive features
    img_features = []
    
    # 2) Apply color conversion if needed
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            # print('converting to YCrCb')
    else:
        feature_image = np.copy(img)

    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    # 4) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    # 5) Compute HOG features if flag is set
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            if vis == True:
                hog_image = []
                for channel in range(feature_image.shape[2]):
                    channel_hog_features, channel_hog_image = get_hog_features(feature_image[:, :, channel], orient,  
                                                                               pix_per_cell, cell_per_block, vis=True)
                    hog_features.append(channel_hog_features)
                    hog_image.append(channel_hog_image)
                hog_image = np.concatenate(hog_image)
            else:
                for channel in range(feature_image.shape[2]):
                    channel_hog_features = get_hog_features(feature_image[:, :, channel], orient, 
                                                            pix_per_cell, cell_per_block, vis=False)
                    hog_features.append(channel_hog_features)

            hog_features = np.ravel(hog_features)     
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                           pix_per_cell, cell_per_block, vis=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False)
        
        img_features.append(hog_features)

    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


def extract_features_from_files(imgs, color_space='RGB', spatial_size=(32, 32),
                                hist_bins=32, orient=9,
                                pix_per_cell=8, cell_per_block=2, hog_channel=0,
                                spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    This function is used for Training the Classifier. 
    It iterates over the location list of images `imgs` and 
    extract features for each one and appends all extraced features
    into a single list of features and returns it.
    
    Inputs
    imgs: file list of images
    color_space: type of space to extract
    spatial_size: down-sample size
    hist_bins: groupings
    orient: # of HOG orientation bins
    pix_per_cell: size in pixels of a cell
    cell_per_block: # of cells in each block
    hog_channel: 'ALL' 3 channels or just the 1 '0 index' channel
    spatial_feat: flag to optionally calc or not
    hist_feat: flag to optionally calc or not
    hog_feat: flag to optionally calc or not
    ---
    Returns 
    list of these feature vectors
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each image, one by one
        image = mpimg.imread(file)
        file_features = extract_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, 
                                         cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        features.append(file_features)

    # Return list of feature vectors
    return features





