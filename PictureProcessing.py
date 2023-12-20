import cv2 as cv
import numpy as np

# img = cv.imread("./pa_logo.png")
img = cv.imread("./golden_gate_bridge.jpg")

# Reshape the image to be a 2D array with 3 channels. 
''' The value -1 the number of rows needed is calculated 
automatically based on the colomns. By reshaping to a 2D array, 
each pixel is a row and each column represents a column (R, G, B).
This allows the k-means cluster algorithm to cluster similar colors
together.'''  
img_reshape = img.reshape((-1, 3))

cv.imshow("Display window", img)
k = cv.waitKey(0)