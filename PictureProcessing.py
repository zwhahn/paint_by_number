import cv2 as cv
import numpy as np

# Load original image
# img = cv.imread("./pa_logo.png")
img = cv.imread("./clifford.jpg")
cv.imshow("Display window", img)  # uncomment to view original image 

# Detect image edges
edges = cv.Canny(img, 100, 200) 
# cv.imshow("Display window 2", edges)

# Overlay edges on original image
edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
overlay_img = img + edges_bgr
# cv.imshow("Image with Edge Detection", overlay_img)

# Lower and higher HSV of red
lower_hsv1 = np.array([0, 100, 100])
upper_hsv1 = np.array([10, 255, 255])

lower_hsv2 = np.array([160, 100, 100])
upper_hsv2 = np.array([179, 255, 255])

# Define color dictionary (credit: Ari Hashemian https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv)
color_dict_HSV = {'black': [[179, 255, 30], [0, 0, 0]],
              'white': [[179, 18, 255], [0, 0, 231]],
              'red1': [[179, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[179, 18, 230], [0, 0, 40]]}


# Generating red mask
red_lower_mask = cv.inRange(img, np.array(color_dict_HSV['red1'][1]), np.array(color_dict_HSV['red1'][0]))
red_upper_mask = cv.inRange(img, np.array(color_dict_HSV['red2'][1]), np.array(color_dict_HSV['red2'][0]))
full_red_mask = red_lower_mask + red_upper_mask

red_detected_img = cv.bitwise_and(img, img, mask = full_red_mask)
cv.imshow("Detected images", red_detected_img)

# Generate black mask
black_mask = cv.inRange(img, np.array(color_dict_HSV["black"][1]), np.array(color_dict_HSV["black"][0]))
black_detected_img = cv.bitwise_and(img, img, mask = black_mask)

cv.imshow("Black Detected Images", black_detected_img)


cv.waitKey(0)  # keep images open until any key is pressed