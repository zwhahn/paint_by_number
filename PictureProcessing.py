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
# Define color dictionary (credit: Ari Hashemian https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv)
color_dict_HSV = {'black': [[179, 255, 30], [0, 0, 0]],
              'white': [[179, 18, 255], [0, 0, 231]],
              'red1': [[179, 255, 255], [150, 30, 30]],
              'red2': [[25, 100, 245], [0, 0, 0]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[135, 255, 255], [90, 0, 0]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[179, 18, 230], [0, 0, 40]]}


# Generating red mask
red_lower_mask = cv.inRange(img, np.array(color_dict_HSV['red2'][1]), np.array(color_dict_HSV['red2'][0]))
red_upper_mask = cv.inRange(img, np.array(color_dict_HSV['red1'][1]), np.array(color_dict_HSV['red1'][0]))
full_red_mask = red_lower_mask + red_upper_mask

red_detected_img = cv.bitwise_and(img_simplified, img_simplified, mask = full_red_mask)
cv.imshow("Red Detected Images", red_detected_img)

# Generate black mask
black_mask = cv.inRange(img_simplified, np.array(color_dict_HSV["black"][1]), np.array(color_dict_HSV["black"][0]))
black_detected_img = cv.bitwise_and(img_simplified, img_simplified, mask = black_mask)

# cv.imshow("Black Detected Images", black_detected_img)

# Generate blue mask
blue_mask = cv.inRange(img, np.array(color_dict_HSV["blue"][1]), np.array(color_dict_HSV["blue"][0]))
blue_detected_img = cv.bitwise_and(img_simplified, img_simplified, mask = blue_mask)

# cv.imshow("Blue Detected Images", blue_detected_img)


cv.waitKey(0)  # keep images open until any key is pressed