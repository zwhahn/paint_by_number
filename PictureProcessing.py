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

# Convert to float32
img_reshape = np.float32(img_reshape)

# Define criteria, number of clusters(K), and apply kmeans()
'''cv.TERM_CRITERIA_EPS indicates that the algorithm should stop when the specified accuracy (epsilon) is reached.
cv.TERM_CRITERIA_MAX_ITER indicates that the algorithm should stop after the specified number of iterations (max_iter) 1. '''
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # stop criteria, epsilon, max_iter
K = 8  # number of clusters (or colors)
ret, label, center = cv.kmeans(img_reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert back to uint8
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow('res2', res2)

cv.imshow("Display window", img)
k = cv.waitKey(0)