import cv2 as cv

# Load original image
img = cv.imread("./pa_logo.png")
# cv.imshow("Display window", img)  # uncomment to view original image 

# Detect image edges
edges = cv.Canny(img, 100, 200) 
cv.imshow("Display window 2", edges)

# Overlay edges on original image
edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # convert edges to bgr
overlay_img = img + edges_bgr
cv.imshow("Image with Edge Detection", overlay_img)


cv.waitKey(0)  # keep images open until any key is pressed