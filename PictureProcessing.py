import cv2 as cv
import numpy as np

# Load original image
img = cv.imread("./pa_logo.png")
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
color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}


# Generating mask
lower_mask = cv.inRange(img, lower_hsv1, upper_hsv1)
upper_mask = cv.inRange(img, lower_hsv2, upper_hsv2)
full_red_mask = lower_mask + upper_mask

detected_img = cv.bitwise_and(img, img, mask = full_red_mask)

cv.imshow("Detected images", detected_img)


cv.waitKey(0)  # keep images open until any key is pressed