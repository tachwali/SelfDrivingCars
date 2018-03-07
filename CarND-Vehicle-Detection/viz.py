####################################
## functions for visualization
####################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    """
    Draw rectangular coloured boxes onto an image.
    img: to draw rectangle onto
    boxes: list of bounding boxes
    color: defaults to blue
    thick: box line thickness
    ---
    Returns
    boxes drawn on input image
    """
    imcopy = np.copy(img)

    # draw each bounding box on your image copy using cv2.rectangle()
    # Iterate through the bounding boxes
    for bbox in boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def visualize(rows, cols, imgs, titles):
    """
    Plots multiple images
    rows: # of image rows
    cols: # of image cols
    imgs: # imgs list
    titles: image titles
    """
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.title(i + 1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
            # cv2.imwrite('CarND-Vehicle-Detection-master/output_images/tmp/'+str(i)+'.jpg', img)  # store img


def draw_labeled_boxes(img, labels):
    """
    Draw bounding box around detected cars in image.
    img: to draw rectangles on detected cars
    labels: detected cars
    drawn on image
    """
    # Iterate through all detected cards
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img



