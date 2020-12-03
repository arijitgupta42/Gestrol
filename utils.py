# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# running average function to calibrate the thresholding 
def run_avg(image, aWeight):
	global bg
	if bg is None:
		bg = image.copy().astype("float")
		return

	cv2.accumulateWeighted(image, bg, aWeight)

# segmentation of the hand
def segment(image, threshold=25):
	global bg
	diff = cv2.absdiff(bg.astype("uint8"), image)

	thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

	cnts,_ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(cnts) == 0:
		return
	else:
		segmented = max(cnts, key=cv2.contourArea)
		return (thresholded, segmented)
