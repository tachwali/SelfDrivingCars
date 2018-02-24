####################################
## function for camera clibration
####################################
import cv2
import numpy as np

def find_corners(img, nx=9, ny=6, draw=True):
    '''
	Generates the corners for chess board with nx columns and ny raws
    Inputs
    img: input chessboard image
    nx: number of inside corners on the x axis
    ny: number of inside corners on the y axis
	draw: if True, the corners will be drawn on the input img
    ---
    Returns
    matrix of corners if there were corners
    '''
    ## convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    ## if found, draw corners
    if ret == True:
        # Draw and display the corners
        if draw:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        return corners

def get_objpoints(nx, ny):    
	'''
	Generates the ideal grid corners for chess board with nx columns and ny raws
	Inputs:
	nx: number of inside corners on the x axis
    ny: number of inside corners on the y axis
	---
	Returns
	Numpy array of shape (nx.ny,3) of indeces for ideal grid corners which 
	can be used as objpoints for cv2.calibrateCamera  
	'''
	nchannels = 3
	objp = np.zeros((nx*ny,nchannels), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)     
	return objp


def undistort_img(img, camera_matrix, distortion_matrix):
    '''
	Undistorts image "img" using clibration matrix "camera_matrix" and Input vector of distortion coefficients "distortion_matrix"
    Inputs
    img: input chessboard image
    camera_matrix: output from OpenCV `calibrateCamera` function
    distortion_matrix: output from OpenCV `calibrateCamera` function
    ---
    Returns
    undistorted image
    '''
    ## using OpenCV's undistortion as before
    undist_img = cv2.undistort(img, camera_matrix, distortion_matrix, None, camera_matrix)
    
    return undist_img
