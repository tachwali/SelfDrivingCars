import matplotlib.pyplot as plt
import numpy as np

def display_image(img, handle=None, label=""):
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
    ## plot the images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    f.tight_layout()
  
    display_image(before, ax1, label1)
    display_image(after, ax2, label2)
