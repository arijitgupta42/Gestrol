'''

Code based on:
https://www.pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
https://gogul.dev/software/hand-gesture-recognition-p1

'''

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import numpy as np
import utils
import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from pynput.keyboard import Key,Controller
keyboard = Controller()

tfmodel = tf.keras.models.load_model('models/CNN.h5')
classes= ['Fist','Open Hand','Two Fingers']

# declaring some global variables
bg = None
utils.bg = bg
num_frames = 0
aWeight = 0.5

# declaring the model to be used for super-resolution
model = "models/FSRCNN_x3.pb"
modelName = model.split(os.path.sep)[-1].split("_")[0].split('/')[-1].lower()
modelScale = model.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# print("[INFO] Model name: {}".format(modelName))
# print("[INFO] Model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model)
sr.setModel(modelName, modelScale)

# starting capture of camera stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
thresholded = imutils.resize(vs.read(), height=250)

# in video stream loop
while True:
	if num_frames == 0:
		print("[INFO] Hold Still, Calibrating Input...")
	# flip frame, create a grayscaled and a cropped version
	frame = cv2.flip(vs.read(),1)
	# frame = vs.read()
	sky = frame[0:250, 0:250]
	gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# upscale the image using super-resolution
	upscaled = imutils.resize(sr.upsample(sky), width = 125)
	sky = imutils.resize(sky, width=400)

	# draw rectangle over the roi where the hand should be
	frame = cv2.rectangle(frame, (0,0), (250,250), (253,71,3), 3)

	# calibrate the background for the threshold function for the first 30 frames
	if num_frames < 30:
		'''
		frame = cv2.putText(frame, 'Hold Still, Calibrating....', (5,15), cv2.FONT_HERSHEY_DUPLEX,  
                   0.5, (0,255,0), 1, cv2.LINE_AA) 
		'''
		utils.run_avg(gray, aWeight)
	else:
		if num_frames == 30:
			'''
			frame = cv2.putText(frame, 'Calibrated!', (5,15), cv2.FONT_HERSHEY_DUPLEX,  
                   0.5, (0,255,0), 1, cv2.LINE_AA) 
			'''	
			print("[INFO] Calibrated! Please keep your hand inside the colored boundary for it to be detected.")

		elif num_frames == 40:
			'''
			frame = cv2.putText(frame, 'You can recalibrate by pressing R', (5,15), cv2.FONT_HERSHEY_DUPLEX,  
                   0.4, (0,255,0), 1, cv2.LINE_AA) 
			'''
			print("[INFO] You can press R to recalibrate")
		hand = utils.segment(gray)
		if hand is not None:
			(thresholded, segmented) = hand

			cv2.drawContours(frame, [segmented], -1, (0, 255, 0))
			frame = imutils.resize(frame, height=125)
			
			# display the segmented image
			thresholded = imutils.resize(thresholded, height=125)
			thresholded = np.stack((thresholded,)*3, axis=-1)
			tfimage=thresholded.reshape(-1,125,125,3)
			'''
			for i in range(125):
				for j in range(125):
					for k in range(3):
						thresholded[i,j,k] = (thresholded[i,j,k]/255)*(gray[i,j]+20)
			'''

			#print(thresholded)
			pred = 0
			if num_frames%2==0:
				pred = (np.argmax(tfmodel.predict(tfimage), axis=-1)[0])
			
			if pred==1:
				keyboard.release(Key.down)
				keyboard.press(Key.space)
				keyboard.release(Key.space)
				print(classes[pred])
			
			elif pred==2:
				keyboard.press(Key.down)
				print(classes[pred])
			
			frame = np.concatenate((frame, thresholded), axis=1)
		
			# cv2.imshow("Input Recieved", thresholded)
			# cv2.moveWindow('Input Recieved',340,0)

	# display actual camera stream
	frame = imutils.resize(frame, height=125)
	frame = np.concatenate((frame, upscaled), axis=1)
	
	cv2.imshow("Live Feed", frame)
	cv2.moveWindow('Live Feed',0,0)

	# dispaly upscaled cropped section
	# cv2.imshow("Cropped Super Resolution", upscaled)

	num_frames += 1

	# exit condition
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	elif key == ord("r"):
		num_frames = 0
		bg = None

cv2.destroyAllWindows()
vs.stop()

