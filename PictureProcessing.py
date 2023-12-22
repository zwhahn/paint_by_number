import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 

# Load original image
# img = cv.imread("./pa_logo.png")
# img = cv.imread("./golden_gate_bridge.jpg")
# img = cv.imread("./clifford.jpg")
# img = cv.imread("./color_circles.jpg")
img = cv.imread("./brad_pitt.jpg")

# Blur image to reduce noise for improved edge detection
img_blur = cv.GaussianBlur(img,(7,7), sigmaX=30, sigmaY=30)


'''COLOR QUANTIZATION'''
# Reshape the image to be a 2D array with 3 channels. 
# The value -1 the number of rows needed is calculated 
# automatically based on the colomns. By reshaping to a 2D array, 
# each pixel is a row and each column represents a column (R, G, B).
# This allows the k-means cluster algorithm to cluster similar colors
# together.  
img_reshape = img_blur.reshape((-1, 3))

# Convert to float32
img_reshape = np.float32(img_reshape)

# Define criteria, number of clusters(K), and apply kmeans()
# cv.TERM_CRITERIA_EPS indicates that the algorithm should stop when the specified accuracy (epsilon) is reached.
# cv.TERM_CRITERIA_MAX_ITER indicates that the algorithm should stop after the specified number of iterations (max_iter) 1.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # stop criteria, epsilon, max iterations
K = 6  # number of clusters (or colors)
ret, label, center = cv.kmeans(img_reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert back to uint8
center = np.uint8(center)  # RGB values of the final clusters
# print(center)
img_simplified = center[label.flatten()]
img_simplified = img_simplified.reshape((img.shape))


'''COLOR MASKING'''
# Compute the upper and lower limits
tol = 5  # tolerance 
bgr_color_limit_dict = {}
for count, bgr_color in enumerate(center):
    bgr_color_limit_dict[count] = np.array([center[count][0] - tol, center[count][1] - tol, center[count][2] - tol]), np.array([center[count][0] + tol, center[count][1] + tol, center[count][2] + tol])

# Create masks
mask_dict = {}
for count, color_limit in enumerate(bgr_color_limit_dict):
    mask_dict[count] = cv.inRange(img_simplified, bgr_color_limit_dict[count][0], bgr_color_limit_dict[count][1]) 

# Apply masks
mask_img_dict = {}
for count, mask in enumerate(mask_dict):
    mask_img_dict[count] = cv.bitwise_and(img_simplified, img_simplified, mask = mask_dict[count])


'''EDGE DETECTION'''
# Detect image edges
edges = cv.Canny(mask_img_dict[0], 100, 200) 

# Overlay edges on original image
edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
overlay_img = img_simplified + edges_bgr


'''SECTIONING'''



'''MULTI DISPLAY'''
# Used method from geeksforgeeks.org (https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/)
fig = plt.figure(figsize=(10,7))

rows = 2
columns = 2

# Add subplot in first position
fig.add_subplot(rows, columns, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original")

# Add subplot in second position
fig.add_subplot(rows, columns, 2)
plt.imshow(cv.cvtColor(img_blur, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Blurred")

# Add subplot in third position
fig.add_subplot(rows, columns, 3)
plt.imshow(cv.cvtColor(mask_img_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1")

# Add subplot in fourth position
fig.add_subplot(rows, columns, 4)
plt.imshow(cv.cvtColor(edges_bgr, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1 Edges")


'''IMAGES TO SHOW'''
# cv.imshow("Original Image", img)
cv.imshow("Blurred Image", img_blur)
cv.imshow("Simplified Image", img_simplified)
# cv.imshow("Simplified Image Edges", edges)
# cv.imshow("Simplified Image overlaid with Edge Detection", overlay_img)
cv.imshow("Mask Image 1", mask_img_dict[0])
cv.imshow("Mask Image 1 Edges", edges_bgr)
# cv.imshow("Mask Image 2", mask_img_dict[1])
# cv.imshow("Mask Image 3", mask_img_dict[2])


# plt.show()
cv.waitKey(0)  # keep images open until any key is pressed