import numpy as np
import cv2


import features as fs
import utils as ut
# =====
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Create a list of windows to be searched on for a given image
    Takes an image shape, start and stop positions in both x and y, window size (x and y dimensions), and overlap
    fraction (for both x and y)
    Inputs
    img: to be searched
    x_start_stop: Region Of Interest in x direction
    y_start_stop: Region Of Interest in y direction
    xy_window: size to draw
    xy_overlap: how much you want the window to overlap with adjacent
    ---
    Returns
    a list of windows, each window is a tuple ((startx, starty), (endx, endy))
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_step) 
    
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """
    Using the trained classifier, search car OR notcar and return windows containing cars.
    Inputs
    img: to run trained classifier over
    windows: to be searched aka output from slide_windows()
    clf: the trained classifier
    scaler: scales extracted features to be fed to classifier
    color_space: type of space to extract
    spatial_size: down-sample size
    hist_bins: number of groupings
    orient: number of HOG orientation bins
    pix_per_cell: size in pixels of a cell
    cell_per_block: number of cells in each block
    hog_channel: 'ALL' 3 channels or just the 1 '0 index' channel
    spatial_feat: flag to optionally calc or not
    hist_feat: flag to optionally calc or not
    hog_feat: flag to optionally calc or not
    ---
    Returns
    list of positive detection windows
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image and resize to the same size as used in Training cutouts
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        features = fs.extract_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        # 5) Scale extracted features to be fed to classifier using scaler that we fit using the Training data
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows



