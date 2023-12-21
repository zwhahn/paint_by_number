import cv2 as cv
import numpy as np

# Load original image
# img = cv.imread("./pa_logo.png")
img = cv.imread("./golden_gate_bridge.jpg")
cv.imshow("Original image", img)

'''COLOR QUANTIZATION'''
# Reshape the image to be a 2D array with 3 channels. 
# The value -1 the number of rows needed is calculated 
# automatically based on the colomns. By reshaping to a 2D array, 
# each pixel is a row and each column represents a column (R, G, B).
# This allows the k-means cluster algorithm to cluster similar colors
# together.  
img_reshape = img.reshape((-1, 3))

# Convert to float32
img_reshape = np.float32(img_reshape)

# Define criteria, number of clusters(K), and apply kmeans()
# cv.TERM_CRITERIA_EPS indicates that the algorithm should stop when the specified accuracy (epsilon) is reached.
# cv.TERM_CRITERIA_MAX_ITER indicates that the algorithm should stop after the specified number of iterations (max_iter) 1.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # stop criteria, epsilon, max_iter
K = 3  # number of clusters (or colors)
ret, label, center = cv.kmeans(img_reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert back to uint8
center = np.uint8(center)  # RGB values of the final clusters
print(center)
img_simplified = center[label.flatten()]
img_simplified = img_simplified.reshape((img.shape))
cv.imshow('Simplified Image', img_simplified)

'''EDGE DETECTION'''
# Detect image edges
edges = cv.Canny(img_simplified, 100, 200) 
cv.imshow("Simplified Image Edges", edges)

# Overlay edges on original image
edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
overlay_img = img_simplified + edges_bgr
# cv.imshow("Simplified Image overlaid with Edge Detection", overlay_img)

'''COLOR MASKING'''
# Compute the upper and lower limits
tol = 10  # toleramce 
bgr_color_limit_dict = {}
for count, bgr_color in enumerate(center):
    bgr_color_limit_dict[count] = np.array([center[count][0] - tol, center[count][1] - tol, center[count][2] - tol]), np.array([center[count][0] + tol, center[count][1] + tol, center[count][2] + tol])

print("bgr_color_limit_dict: ", bgr_color_limit_dict)

# Create masks
mask_dict = {}
for count, color_limit in enumerate(bgr_color_limit_dict):
    mask_dict[count] = cv.inRange(img_simplified, bgr_color_limit_dict[count][0], bgr_color_limit_dict[count][1]) 
    
mask = cv.inRange(img_simplified, np.array([center[1][0] - 10, center[1][1] - 10, center[1][2] - 10]), np.array([center[1][0] + 10, center[1][1] + 10, center[1][2] + 10]))
result = cv.bitwise_and(img_simplified, img_simplified,  mask = mask)

cv.imshow("Mask Images", result)
# print(type(img_simplified))

cv.waitKey(0)  # keep images open until any key is pressed