import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import imutils

# Load original image
# img = cv.imread("./pa_logo.png")
# img = cv.imread("./golden_gate_bridge.jpg")
img = cv.imread("./clifford.jpg")
# img = cv.imread("./color_circles.jpg")
# img = cv.imread("./brad_pitt.jpg")

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
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # print("hierarchy test: ", hierarchy[0])
    contours = imutils.grab_contours([contours, hierarchy])  # Extract contours and returns them as a list. Output of cv.findContours can be different depending on version being used

    # Process and draw contours
    for count, contour in enumerate(contours):
        # Compute the center
        M = cv.moments(contour)
        if int(M["m00"]) != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        
            cv.drawContours(input_img_copy, [contour], -1, (0, 255, 0), 2)

            # # Only draw contours that don't have children
            # if hierarchy[0][count][3] == -1 or hierarchy[0][count][3] != -1:
            #     # Draw contour and center on image
            #     cv.circle(input_img_copy, (center_x, center_y), 7, (255, 255, 255), -1)
            #     # cv.putText(input_img_copy, "center", (center_x - 20, center_y - 20), 
            #             # cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return contours, input_img_copy

mask_img_cntr_dict = {}
cntr_dict = {}  # dict with a list of numpy arrays, each array represents a contour
for count, mask_img in enumerate(mask_img_dict):
    contours, output_img = contour_func(mask_img_dict[mask_img])
    mask_img_cntr_dict[count] = output_img
    cntr_dict[count] = contours

'''LABELING'''
# Following method from openCV docs (https://docs.opencv.org/3.4/dc/d48/tutorial_point_polygon_test.html)

img_copy = mask_img_cntr_dict[0].copy()
grayscale_image_for_label_func = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
# Loop through all pixels in img and calculate distances to the contour, positive value means its inside of contour
def label_func(contour, grayscale_image_for_label_func = grayscale_image_for_label_func):
    raw_dist = np.empty(grayscale_image_for_label_func.shape, dtype=np.float32)  # initialize numpy array for each pixel in img
    for i in range(grayscale_image_for_label_func.shape[0]): 
        for j in range(grayscale_image_for_label_func.shape[1]):
            if cv.pointPolygonTest(contour, (j, i), False) > 0:  # check if point is inside contour
                # print("Inside contour")
                raw_dist[i,j] = cv.pointPolygonTest(contour, (j,i), True)  # calculate distance  
                minVal, maxVal, _, maxLoc = cv.minMaxLoc(raw_dist)  # calculate max location (maxLoc)
                return maxLoc

# print("cntr_dict: ", len(cntr_dict))
# print(cntr_dict[0])
# Loop through all contours, find maxLoc and save to dictionary
maxLoc_dict = {}
for _, contours in cntr_dict.items():
    # print("contours: ", contours)
    for count, contour in enumerate(contours):
        x, y, z = contour.shape
        # Only use the larger contours
        if x > 100:
            # print("contour: ", type(contour))
            # print("contour shape: ", contour.shape)
            maxLoc_dict[count] = label_func(contour)

# print("contourS: ", type(contours))

# Loop through all maxLoc and draw a circle there
for count, location in enumerate(maxLoc_dict):
    print("maxLoc length: ", len(maxLoc_dict))
    cv.circle(mask_img_cntr_dict[0], maxLoc_dict[location], 7, (0, 0, 255), -1)
    cv.putText(mask_img_cntr_dict[0], str(count+1), (maxLoc_dict[location][0] - 20, maxLoc_dict[location][1] - 20),  
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


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
# cv.imshow("Test Image", img_test)


cv.waitKey(0)  # keep images open until any key is pressed