from scipy.ndimage import measurements 
import numpy as np
import cv2

import features as fs
import viz
import utils as ut
# =====
def add_heat(heatmap, bbox_list):
    """
    Create heatmap in bounding boxes
    :param heatmap: of detections
    :param bbox_list: bounding boxes
    :return: updated heatmap
    """
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


# =====
def apply_threshold(heatmap, threshold):
    """
    Remove false-positive detections by replacing pixel values with 0 if fall below threshold
    :param heatmap: detections
    :param threshold: to trigger setting 0 value
    :return: heatmap with false-positive car detections removed (hopefully)
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


class car_finder():
    def __init__(self, scale, y_start_stop, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, X_scaler, classifier):
        self.scale = scale
        self.y_start_stop = y_start_stop 
        self.orient = orient 
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block 
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.X_scaler = X_scaler
        self.classifier = classifier

    def find_cars(self, img):
        """
        Find cars either in individual windows or across the entire image if scale != 1.
        Inputs
        img: image (or individual windows) to apply Classifier on
        scale: 1 == classify on windows, 1 < scale > 1 == classify on entire image
        y_start_stop: Region Of Interest in y direction
        orient: number of HOG orientation bins
        pix_per_cell: size in pixels of a cell
        cell_per_block: number of cells in each block
        spatial_size: down-sample size
        hist_bins: Number of histogram bins
        X_scaler: StandardScaler object used to scale extracted features in training samples
        classifier: classifier object trained to detect vehicles
        ---
        Returns
        drawn on image, detection boxes and heatmaps
        """
        img_boxes = []  # Clears img_boxes so we don't keep unwanted heatmap history
        count = 0
        draw_img = np.copy(img)

        ystart = self.y_start_stop[0]
        ystop = self.y_start_stop[1]

        # Make a heatmap of zeros
        heatmap = np.zeros_like(img[:, :, 0])

        # IMPORTANT : reading *.jpeg's (scaled 0-255, aka scaling needed), but
        # # trained on *.png's (scaled 0-1, aka scaling not needed)
        if img.dtype == 'uint8':
            img = img.astype(np.float32) / 255  # aka scaling needed

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = ut.convert_color(img_tosearch, conv='RGB2YCrCb')

        if self.scale != 1:  # resize whole image instead of separate windows
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / self.scale), np.int(imshape[0] / self.scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        # These hold the number of HOG cells

        nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1  # Note : '//' causes integers to be result, instead of floats
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1

        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - 1

        # aka 75% overlap between cells
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = fs.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = fs.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = fs.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get colour features
                spatial_features = fs.bin_spatial(subimg, size=self.spatial_size)
                hist_features = fs.color_hist(subimg, nbins=self.hist_bins)
                # Scale features and make a prediction
                stacked_feature = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.X_scaler.transform(stacked_feature)
                test_prediction = self.classifier.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * self.scale)
                    ytop_draw = np.int(ytop * self.scale)
                    win_draw = np.int(window * self.scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255))
                    img_boxes.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                    heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left:xbox_left + win_draw] += 1

        return draw_img, img_boxes, heatmap

    def process_image(self, img):
        out_img, out_boxes, heat_map = self.find_cars(img) 
        labels = measurements.label(heat_map)
        # Draw bounding boxes on a copy of the image
        draw_img = viz.draw_labeled_boxes(np.copy(img), labels)
        return draw_img

