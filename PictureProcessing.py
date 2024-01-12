import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import imutils
import time

# Timer Start
start_time = time.time()

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
# The value -1 the number of rows needed is calculated automatically based on the colomns. By reshaping to a 2D array, 
# each pixel is a row and each column represents a column (R, G, B).
# This allows the k-means cluster algorithm to cluster similar colors together.  
img_reshape = img_blur.reshape((-1, 3))

# Convert to float32 for floating-point calculations
img_reshape = np.float32(img_reshape)

# Define criteria, number of clusters(K), and apply kmeans()
# cv.TERM_CRITERIA_EPS indicates that the algorithm should stop when the specified accuracy (epsilon) is reached.
# cv.TERM_CRITERIA_MAX_ITER indicates that the algorithm should stop after the specified number of iterations (max_iter) 1.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # stop criteria, epsilon, max iterations
K = 16  # number of clusters (or colors)
ret, label, base_colors = cv.kmeans(img_reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

base_colors = np.uint8(base_colors)  # BGR values of the final clusters
# print(base_colors)
img_simplified = base_colors[label.flatten()]  # Replace each picel with its corresponding base color
img_simplified = img_simplified.reshape((img.shape))


'''COLOR MASKING'''
# For each base_color, calculate max and min values to use as mask 
tol = 5  # tolerance 
bgr_color_limit_dict = {}
for i, bgr_color in enumerate(base_colors):
    b_val = base_colors[i][0]
    g_val = base_colors[i][1]
    r_val = base_colors[i][2]
    bgr_color_limit_dict[i] = np.array([b_val - tol, g_val - tol, r_val - tol]), np.array([b_val + tol, g_val + tol, r_val + tol])

# Create masks
mask_dict = {}
for i, color_limit in bgr_color_limit_dict.items():
    # Each pixel that falls in the color range is set to white (255), the rest are set to black (0)
    mask_dict[i] = cv.inRange(img_simplified, bgr_color_limit_dict[i][0], bgr_color_limit_dict[i][1]) 

# Apply masks
img_mask_dict = {}
for i, mask in mask_dict.items():
    # Keeps the pixel values from img_simplified where the mask is white
    img_mask_dict[i] = cv.bitwise_and(img_simplified, img_simplified, mask = mask_dict[i])


'''FIND CONTOURS'''
def find_contours(img_mask, img_thresh):
    # Following method from pyimagesearch.com (https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours([contours, hierarchy])  # Extract contours and returns them as a list. Output of cv.findContours can be different depending on version being used
    return hierarchy, contours

hierarchy_dict = {}  # Contour hierarchy information
cntr_dict = {}  # Values are lists of numpy arrays, each array represents a contour
for i, img_mask in img_mask_dict.items():
    hierarchy, contours = find_contours(img_mask, mask_dict[i])
    cntr_dict[i] = contours
    hierarchy_dict[i] = hierarchy


'''LABELING'''
# Constants
img_size = img.shape[:2]  # Columns and rows
area_limit = 500  # Don't label feature that is too small
width_limit = 10  # Don't label feature that is too thin
border_size = 1

# Add border to ensure distanceTransform recognizes edge of photo
def add_border(img, border_size=border_size):
    img_border = cv.copyMakeBorder(img, 
                                   top= border_size,
                                   bottom= border_size,
                                   left= border_size,
                                   right= border_size,
                                   borderType= cv.BORDER_CONSTANT,
                                   value= [0, 0, 0])
    return img_border

# Check if blob is on the edge of the image
def blob_is_on_image_edge(x_pos, y_pos, width, height, img_size = img_size):
    if x_pos == 0 or y_pos == 0 or (x_pos + width) == img_size[1] or (y_pos + height) == img_size[0]:
        return True
    else:
        return False  


def find_label_location(blob):
    # Calculate shortest distance from each white to a black pixel
    dist_transform = cv.distanceTransform(blob, cv.DIST_L2, 3)

    # Return the largest distance and location of the pixel (most space for a clear number label)
    _,max_val,_, max_loc = cv.minMaxLoc(dist_transform)
    return max_val, max_loc


def draw_empty_contours_and_labels(mask):
    label_location_list = []  # Initialize list to store label locations

    # Find all 'blobs' in threshold image
    (total_labels, label_ids, stats, _) = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)

    # Initialize empty mask
    empty_contour = np.zeros(mask.shape, dtype="uint8")

    for blob_id in range(1,total_labels):
        x_pos, y_pos, width, height, area = stats[blob_id]
        blob = (label_ids == blob_id).astype("uint8") * 255  # draw blob in white

        # If the blob is on the edge of the image, add a border to distanceTransform will recognize edge
        if blob_is_on_image_edge(x_pos, y_pos, width, height):
            blob_with_border = add_border(blob)
            max_val, max_loc = find_label_location(blob_with_border)
        
        else:
            max_val, max_loc = find_label_location(blob)

        # If the blob area and the distance from any wall is large enouig
        if area > area_limit and max_val > width_limit:
            empty_contour = cv.bitwise_or(empty_contour, blob)
            if max_val > width_limit:
                label_location_list.append(max_loc)
    return empty_contour, label_location_list

label_locations_dict = {}
empty_contours_dict = {}
for i, mask in mask_dict.items():
    empty_contour, label_location_list = draw_empty_contours_and_labels(mask)
    empty_contours_dict[i] = empty_contour
    label_locations_dict[i] = label_location_list


'''COMBINING IMAGES'''
def blend_mask_and_contours(mask, contour_image):
    three_channel_thresh_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    blended = cv.addWeighted(three_channel_thresh_image, 1, contour_image, 1, 0)
    return blended

blended_img_dict = {}
for i, empty_contours in empty_contours_dict.items():
    blended_img_dict[i] = blend_mask_and_contours(empty_contours, img_mask_dict[i])


def combine_all(previous_image, current_image):
    final_image = cv.addWeighted(previous_image, 1, current_image, 1, 0)
    return final_image

for i, blended_image in blended_img_dict.items():
    if i == 0:
        final_image = blended_image
    else:
        previous_image = final_image
        current_image = blended_image
        final_image = combine_all(previous_image, current_image)


# Draw all contour outlines (for coloring in)
for i, contour_list in cntr_dict.items():
    for j, contour in enumerate(contour_list):
        final_image = cv.drawContours(final_image, [contour], -1, (0,0,0), 1)


# Label with corresponding numbers
font = cv.FONT_HERSHEY_COMPLEX
font_scale = 0.6
font_thickness = 1
font_color = (0,0,0)
for color_number, label_location_list in label_locations_dict.items():
    for label_location in label_location_list:
        # If area is filled with color, don't label
        y_pos = int(label_location[1])
        x_pos = int(label_location[0])
        b_color = img_mask_dict[color_number][y_pos, x_pos, 0]
        g_color = img_mask_dict[color_number][y_pos, x_pos, 1]
        r_color = img_mask_dict[color_number][y_pos, x_pos, 2]
        if b_color != 0 or g_color != 0 or r_color != 0:
            text_size = cv.getTextSize(str(color_number + 1), font, font_scale, font_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            label_location_circ = (x_pos, y_pos)
            label_location = (int(x_pos - (text_width/2)), int(y_pos + (text_height/2)))
            cv.putText(final_image, str(color_number + 1), label_location, font, font_scale, font_color, font_thickness)
            cv.circle(final_image, label_location_circ, 3, (0,0,255), -1)  # Highlight label location (uncomment to check placement)


'''DISPLAY'''
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
plt.imshow(cv.cvtColor(img_mask_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1")

# Add subplot in fourth position
fig.add_subplot(rows, columns, 4)
plt.imshow(cv.cvtColor(img_mask_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1 w/ Contour")

# plt.show()  # display matplotlib figures 


'''IMAGES TO SHOW'''
# cv.imshow("Original Image", img)
# cv.imshow("Blurred Image", img_blur)
# cv.imshow("Simplified Image", img_simplified)
# cv.imshow("Simplified Image Edges", edges)
# cv.imshow("Simplified Image overlaid with Edge Detection", overlay_img)
# cv.imshow("Mask Image 1", img_mask_dict[0])
# cv.imshow("Mask Image 1 Gray Scale", img_gray)
# cv.imshow("Mask Image 1 Threshold", img_thresh)
# cv.imshow("Mask Image 2", img_mask_dict[1])
# cv.imshow("Mask Image 3", img_mask_dict[2])

# Timer End
end_time = time.time()
total_time = (end_time-start_time)
print(f'Run Time: {total_time:.4f} seconds')

cv.imshow("Final Image", final_image)
cv.waitKey(0)  # keep images open until any key is pressed