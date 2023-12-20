import cv2 as cv
import numpy as np

img = cv.imread("./pa_logo.png")

cv.imshow("Display window", img)
k = cv.waitKey(0)