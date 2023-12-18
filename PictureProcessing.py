import cv2 as cv
from matplotlib import pyplot as plt

# Load original image
img = cv.imread("./pa_logo.png")

cv.imshow("Display window", img)

edges = cv.Canny(img, 100, 200)


plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(122)
plt.imshow(edges)
plt.title("Edge Image")

plt.show()