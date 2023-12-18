import cv2 as cv

# Load original image
img = cv.imread("./pa_logo.png")
# cv.imshow("Display window", img)  # uncomment to view original image 

# Detect image edges
edges = cv.Canny(img, 100, 200)
cv.imshow("Display window 2", edges)
cv.waitKey(0)