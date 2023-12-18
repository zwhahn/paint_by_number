import cv2 as cv
import numpy as np

# Load original image
img = cv.imread("./pa_logo.png")
# cv.imshow("Display window", img)  # uncomment to view original image 

# Detect image edges
edges = cv.Canny(img, 100, 200) 
# cv.imshow("Display window 2", edges)

# Overlay edges on original image
edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
overlay_img = img + edges_bgr
cv.imshow("Image with Edge Detection", overlay_img)

# Lower and higher HSV of red
lower_hsv1 = np.array([0, 10, 25])
upper_hsv1 = np.array([10, 255, 255])

lower_hsv2 = np.array([120, 10, 20])
upper_hsv2 = np.array([179, 255, 255])

# Generating mask
lower_mask = cv.inRange(img, lower_hsv1, upper_hsv1)
upper_mask = cv.inRange(img, lower_hsv2, upper_hsv2)
full_red_mask = lower_mask + upper_mask

detected_img = cv.bitwise_and(img, img, mask = full_red_mask)

cv.imshow("Detected images", detected_img)


cv.waitKey(0)  # keep images open until any key is pressed