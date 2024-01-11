import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import imutils
import time

# Timer Start
start_time = time.time()

# Global Constants
font = cv.FONT_HERSHEY_COMPLEX
fontScale = 0.6

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
K = 6  # number of clusters (or colors)
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


'''CONTOURS'''
def find_contours(img_mask, img_thresh):
    img_mask_and_cntr = img_mask.copy()  # Copy image to not override original
    # Following method from pyimagesearch.com (https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours([contours, hierarchy])  # Extract contours and returns them as a list. Output of cv.findContours can be different depending on version being used

    # Find and Draw contours
    for contour in contours:
        # Compute the center
        M = cv.moments(contour)
        area = int(M["m00"])
        if area > 0:  # only if there is an area     
            cv.drawContours(img_mask_and_cntr, [contour], -1, (0, 0, 0), 1)
    return hierarchy, contours, img_mask_and_cntr

img_mask_and_cntr_dict = {}  # Image with mask applied and contours drawn
hierarchy_dict = {}  # Contour hierarchy information
cntr_dict = {}  # Values are lists of numpy arrays, each array represents a contour
for i, img_mask in img_mask_dict.items():
    hierarchy, contours, img_mask_and_cntr = find_contours(img_mask, mask_dict[i])
    img_mask_and_cntr_dict[i] = img_mask_and_cntr
    cntr_dict[i] = contours
    hierarchy_dict[i] = hierarchy


'''LABELING'''
# Add border to ensure distanceTransform recognizes edge of photo
border_size = 1
def add_border(img, border_size=border_size):
    img_border = cv.copyMakeBorder(img, 
                                   top= border_size,
                                   bottom= border_size,
                                   left= border_size,
                                   right= border_size,
                                   borderType= cv.BORDER_CONSTANT,
                                   value= [0, 0, 0])
    return img_border

img_size = img.shape[:2]  # Columns and rows
area_limit = 500  # Don't label feature that is too small
width_limit = 10  # Don't label feature that is too thin
def find_label_locations(contours, hierarchy, img_size = img_size):
    max_loc_list = []
    for i, contour in enumerate(contours):
        M = cv.moments(contour)  # calculate shape attributes  
        area = int(M["m00"])
        if area > area_limit:  # check area is big enough
            temp = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)  # initialize mask (blank image)
            cv.drawContours(temp, [contour], -1, (0,255,0), cv.FILLED)  # draw larger, filled contour
            
            # Exclude holes by drawing child contours in black
            for j, child_contour in enumerate(contours): # loop through contours again
                if hierarchy[0][j][3] == i:  # if parent is current contour then draw it in black
                    cv.drawContours(temp, [child_contour], -1, (0), thickness=cv.FILLED)
            
            # Add border
            temp_border = add_border(temp)

            # Convery image to correct format for distanceTransform
            gray = cv.cvtColor(temp_border, cv.COLOR_BGR2GRAY)
            bw = cv.convertScaleAbs(gray)

            # Calculate distance from each non-zero pixel to nearest zero pixel
            dist_transform = cv.distanceTransform(bw, cv.DIST_L2, 3)
            _,max_val,_, max_loc = cv.minMaxLoc(dist_transform)  # Find pixel with largest distance to any border (most space for labeling)
            
            # The contour might have a large enough area but is long and skinny. If there is no area that is large enough for a clear label, skip it
            if max_val > width_limit:
                max_loc_list.append(max_loc)

    return max_loc_list

# label_locations_dict = {}
# for i, contours in cntr_dict.items():
#     label_locations_dict[i] = find_label_locations(contours, hierarchy_dict[i])

def draw_empty_contours(mask):
    max_loc_list = []
    # Find all 'blobs' in threshold image
    (total_labels, label_ids, stats, centroid) = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)
    # Initialize empty mask
    empty_contour = np.zeros(mask.shape, dtype="uint8")
    # Go through all 'blobs', if they are larger than the area limit draw them
    for j in range(1,total_labels):
        area = stats[j, cv.CC_STAT_AREA]
        dist_transform = cv.distanceTransform((label_ids == j).astype("uint8") * 255, cv.DIST_L2, 3)
        _,max_val,_, max_loc = cv.minMaxLoc(dist_transform)
        if area > area_limit and max_val > width_limit:
            # cv.circle(final_mask_dict[i], (label_location[0]-border_size, label_location[1]-border_size), 7, (0, 0, 0), -1)
            component_mask = (label_ids == j).astype("uint8") * 255  # draw them in white
            empty_contour = cv.bitwise_or(empty_contour, component_mask)
            if max_val > width_limit:
                max_loc_list.append(max_loc)
    return empty_contour, max_loc_list

label_locations_dict = {}
empty_contours_dict = {}
for i, mask in mask_dict.items():
    empty_contour, max_loc_list = draw_empty_contours(mask)
    empty_contours_dict[i] = empty_contour
    label_locations_dict[i] = max_loc_list



def blend_mask_and_contours(mask, contour_image):
    three_channel_thresh_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    blended = cv.addWeighted(three_channel_thresh_image, 1, contour_image, 1, 0)
    return blended

blended_img_dict = {}
for i, empty_contours in empty_contours_dict.items():
    blended_img_dict[i] = blend_mask_and_contours(empty_contours, img_mask_dict[i])


'''COMBINE ALL IMAGES'''
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
for color_number, label_location_list in label_locations_dict.items():
    for label_location in label_location_list:
        # If area is filled with color, don't label
        y_pos = label_location[1]
        x_pos = label_location[0]
        b_color = img_mask_dict[color_number][y_pos, x_pos, 0]
        g_color = img_mask_dict[color_number][y_pos, x_pos, 1]
        r_color = img_mask_dict[color_number][y_pos, x_pos, 2]
        if b_color != 0 or g_color != 0 or r_color != 0:
            text_size = cv.getTextSize(str(i), font, 1, 1)
            text_width = text_size[0][0]
            text_height = text_size[0][1]
            label_location_circ = (int(x_pos - (border_size)), int(y_pos- (border_size)))
            label_location = (int(x_pos- (border_size + (text_width/2))), int(y_pos- (border_size - (text_height/2))))
            cv.putText(final_image, str(color_number), label_location, font, fontScale, (0,0,0), 1)
            # cv.circle(final_image, label_location_circ, 7, (0,0,255), -1)  # Highlight label location (uncomment to check placement)

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
plt.imshow(cv.cvtColor(img_mask_dict[0], cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mask 1")

# Add subplot in fourth position
fig.add_subplot(rows, columns, 4)
plt.imshow(cv.cvtColor(img_mask_and_cntr_dict[0], cv.COLOR_BGR2RGB))
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
# cv.imshow("Mask Image 1 w/ Contour", img_mask_and_cntr_dict[5])
# cv.imshow("Mask Image 2", img_mask_dict[1])
# cv.imshow("Mask Image 3", img_mask_dict[2])

# Timer End
end_time = time.time()
total_time = (end_time-start_time)
print(f'Run Time: {total_time:.4f} seconds')

cv.imshow("Final Image", final_image)
cv.waitKey(0)  # keep images open until any key is pressed