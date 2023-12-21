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
hsv_color_dict = {} 
def bgr_to_hsv(color):
    bgr_color = np.uint8([[color]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)
    return hsv_color

# Loop through the k-means colors and convert to hsv for masking
for count, bgr_color in enumerate(center):
    print("Print each color:", count, bgr_color)
    print(bgr_to_hsv(bgr_color))
    hsv_color_dict[count] = bgr_to_hsv(bgr_color)

# Compute the upper and lower limits
hsv_color_limit_dict = {}
for count, hsv_color in enumerate(hsv_color_dict):
    lower_limit = hsv_color_dict[count][0][0][0] - 10, 100, 100
    print("lower_limit: ", lower_limit)
    upper_limit = hsv_color_dict[count][0][0][0] + 10, 255, 255
    hsv_color_limit_dict[count] = [upper_limit, lower_limit] 
    
print("hsv_color_limit_dict: ", hsv_color_limit_dict)
# print(type(np.array(hsv_color_limit_dict[1][1])))
# print(type(np.array([15,50,180])))
print(center[0][0])

# hsv_img = cv.cvtColor(img_simplified, cv.COLOR_BGR2HSV)
# mask = cv.inRange(img_simplified, np.array([10,20,20]), np.array([25, 255, 255]))
mask = cv.inRange(img_simplified, np.array([center[1][0] - 10, center[1][1] - 10, center[1][2] - 10]), np.array([center[1][0] + 10, center[1][1] + 10, center[1][2] + 10]))
result = cv.bitwise_and(img_simplified, img_simplified,  mask = mask)

cv.imshow("Mask Images", result)
# print(type(img_simplified))

cv.waitKey(0)  # keep images open until any key is pressed