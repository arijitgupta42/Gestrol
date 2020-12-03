import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import os

img = cv.imread("img.png")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imshow('old', img)
cv.imshow('new', new_img)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=1)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.9*dist_transform.min(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

img = unknown+sure_fg
img = 255-img
cv.imwrite('color_img.jpg', img)
cv.imshow("image", img)
cv.waitKey()