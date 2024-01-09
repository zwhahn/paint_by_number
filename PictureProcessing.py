import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import imutils

# Load original image
# img = cv.imread("./pa_logo.png")
# img = cv.imread("./golden_gate_bridge.jpg")
# img = cv.imread("./clifford.jpg")
# img = cv.imread("./color_circles.jpg")
img = cv.imread("./brad_pitt.jpg")

# Threshold the image to binary using Otsu's method
_, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV)

# Blur image to reduce noise for improved edge detection
img_blur = cv.GaussianBlur(img,(7,7), sigmaX=30, sigmaY=30)

area_limit = 200

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
ret, label, base_colors = cv.kmeans(img_reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert back to uint8
base_colors = np.uint8(base_colors)  # BGR values of the final clusters
# print(base_colors)
img_simplified = base_colors[label.flatten()]
img_simplified = img_simplified.reshape((img.shape))


'''COLOR MASKING'''
# Compute the upper and lower limits
tol = 5  # tolerance 
bgr_color_limit_dict = {}
for count, bgr_color in enumerate(base_colors):
    bgr_color_limit_dict[count] = np.array([base_colors[count][0] - tol, base_colors[count][1] - tol, base_colors[count][2] - tol]), np.array([base_colors[count][0] + tol, base_colors[count][1] + tol, base_colors[count][2] + tol])

# Create masks
mask_dict = {}
for count, color_limit in enumerate(bgr_color_limit_dict):
    mask_dict[count] = cv.inRange(img_simplified, bgr_color_limit_dict[count][0], bgr_color_limit_dict[count][1]) 

# Apply masks
mask_img_dict = {}
for count, mask in enumerate(mask_dict):
    mask_img_dict[count] = cv.bitwise_and(img_simplified, img_simplified, mask = mask_dict[count])


# '''EDGE DETECTION'''
# # Detect image edges
# edges = cv.Canny(mask_img_dict[0], 100, 200) 

# # Overlay edges on original image
# edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
# overlay_img = img_simplified + edges_bgr


'''CONTOURS'''
def contour_func(input_img):
    input_img_copy = input_img.copy()
    # Following method from pyimagesearch.com (https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
    img_gray = cv.cvtColor(input_img_copy, cv.COLOR_BGR2GRAY)  # convert to grayscale
    img_thresh = cv.threshold(img_gray, 60, 255, cv.THRESH_BINARY)[1] 

    # Find contours
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print("hierarchy test: ", hierarchy[0])
    contours = imutils.grab_contours([contours, hierarchy])  # Extract contours and returns them as a list. Output of cv.findContours can be different depending on version being used

    # Process and draw contours
    for count, contour in enumerate(contours):
        # Compute the center
        M = cv.moments(contour)
        if int(M["m00"]) > 0:  # only if there is an area     
            cv.drawContours(input_img_copy, [contour], -1, (0, 255, 0), 1)
    return hierarchy, contours, input_img_copy

mask_img_cntr_dict = {}
hierarchy_dict = {}
cntr_dict = {}  # dict with a list of numpy arrays, each array represents a contour
for count, mask_img in enumerate(mask_img_dict):
    hierarchy, contours, output_img = contour_func(mask_img_dict[mask_img])
    mask_img_cntr_dict[count] = output_img
    cntr_dict[count] = contours
    hierarchy_dict[count] = hierarchy


'''LABELING'''
img_copy = mask_img_cntr_dict[0].copy()
img_size = img.shape[:2]  # only need the columns and rows

def contour_family_label(contours, hierarchy, img_size = img_size):
    max_loc_list = []
    # Loop through each contour
    for i, contour in enumerate(contours):
        M = cv.moments(contour)  # calculate shape attributes  
        area = int(M["m00"])
        if area > area_limit:  # check if contour has no parent and area is big enough
            mask = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)  # initialize mask (blank image)
            cv.drawContours(mask, [contour], -1, (0,255,0), cv.FILLED)  # draw larger, filled contour
            
            # Exclude holes by drawing child contours in black
            for j, child_contour in enumerate(contours): # loop through contours again
                if hierarchy[0][j][3] == i:  # if parent is current contour then draw it in black
                    cv.drawContours(mask, [child_contour], -1, (0), thickness=cv.FILLED)
            
            # Convery image to correct format for distanceTransform
            gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            out = cv.convertScaleAbs(gray)
            # cv.imshow("mask", out)
            # cv.waitKey(0)
            
            dist_transform = cv.distanceTransform(out, cv.DIST_L2, 3)
            _,_,_, max_loc = cv.minMaxLoc(dist_transform)
            max_loc_list.append(max_loc)

    return max_loc_list

label_locations_dict = {}
# Loop through all contours
for i, contours in cntr_dict.items():
    label_locations_dict[i] = contour_family_label(contours, hierarchy_dict[i])

# Loop through max_loc positions and mark them
for i, label_location_list in enumerate(label_locations_dict.items()):
    for label_location in label_location_list[1]:
        # Check if location is black or not, if black don't mark
        b_color = mask_img_cntr_dict[i][label_location[1], label_location[0], 0]
        g_color = mask_img_cntr_dict[i][label_location[1], label_location[0], 1]
        r_color = mask_img_cntr_dict[i][label_location[1], label_location[0], 2]
        if b_color + g_color + r_color != 0:
            cv.circle(mask_img_cntr_dict[i], label_location, 7, (0, 0, 255), -1)


'''MULTI DISPLAY'''
# Used method from geeksforgeeks.org (https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/)
fig = plt.figure(figsize=(10,7))

rows = 2
columns = 2

# Add subplot in first position
fig.add_subplot(rows, columns, 1)
plt.imshow(cv.cvtColor(img_blur, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original Blurred")

# Add subplot in second position
fig.add_subplot(rows, columns, 2)
plt.imshow(cv.cvtColor(img_simplified, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Simplified Image")

# Add subplot in third position
fig.add_subplot(rows, columns, 3)
plt.imshow(cv.cvtColor(mask_img_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1")

# Add subplot in fourth position
fig.add_subplot(rows, columns, 4)
plt.imshow(cv.cvtColor(mask_img_cntr_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1 w/ Contour")

# plt.show()  # display matplotlib figures 


'''IMAGES TO SHOW'''
# cv.imshow("Original Image", img)
# cv.imshow("Blurred Image", img_blur)
# cv.imshow("Simplified Image", img_simplified)
# cv.imshow("Simplified Image Edges", edges)
# cv.imshow("Simplified Image overlaid with Edge Detection", overlay_img)
# cv.imshow("Mask Image 1", mask_img_dict[0])
# cv.imshow("Mask Image 1 Gray Scale", img_gray)
# cv.imshow("Mask Image 1 Threshold", img_thresh)
cv.imshow("Mask Image 1 w/ Contour", mask_img_cntr_dict[0])
# cv.imshow("Mask Image 2", mask_img_dict[1])
# cv.imshow("Mask Image 3", mask_img_dict[2])


cv.waitKey(0)  # keep images open until any key is pressed