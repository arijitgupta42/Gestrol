'''

Code based on:
https://www.pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
https://gogul.dev/software/hand-gesture-recognition-p1

'''

# import the necessary packages
from imutils.video import VideoStream
from keyboard import map_key
import argparse
import imutils
import time
import cv2
import os
import numpy as np
import utils
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from pynput.keyboard import Key,Controller
keyboard = Controller()

tfmodel = tf.keras.models.load_model('models/CNN.h5')
classes= ['Fist','Open Hand', 'OK', "Thumbs Down", "Thumbs Up"]

# declaring some global variables
bg = None
utils.bg = bg
num_frames = 0
aWeight = 0.35

# Map the keys that you wish to control
print("[INFO] Enter the number of keys you wish to map (Max 5)")
num_keys = int(input())
classes = classes[:num_keys]
keys=[]
for gesture in classes:
	print("[INFO] Enter the key you wish to map to the {} gesture".format(gesture))
	key = map_key()
	keys.append(key)
	print("The key mapped was {}".format(key))
	
# declaring the model to be used for super-resolution
model = "models/FSRCNN_x3.pb"
modelName = model.split(os.path.sep)[-1].split("_")[0].split('/')[-1].lower()
modelScale = model.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])
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
	upscaled = imutils.resize(sr.upsample(sky), width = 250)
	gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# draw rectangle over the roi where the hand should be
	frame = cv2.rectangle(frame, (0,0), (250,250), (253,71,3), 3)

	# calibrate the background for the threshold function for the first 30 frames
	if num_frames < 30:
		utils.run_avg(gray, aWeight)
	else:
		if num_frames == 30:
			print("[INFO] Calibrated! Please keep your hand inside the colored boundary for it to be detected.")

		elif num_frames == 40:
			print("[INFO] You can press R to recalibrate")

		hand = utils.segment(gray)
		if hand is not None:
			(thresholded, segmented) = hand

			cv2.drawContours(frame, [segmented], -1, (0, 255, 0))
		
			# display the segmented image
			thresholded = imutils.resize(thresholded, height=125)
			thresholded = np.stack((thresholded,)*3, axis=-1)
			tfimage=cv2.flip(thresholded,1).reshape(-1,125,125,3)

			try:
				temp = pred
			except:
				temp = (np.argmax(tfmodel.predict(tfimage), axis=-1)[0])
			pred = (np.argmax(tfmodel.predict(tfimage), axis=-1)[0])
			message = "No Recognizable Command"
			scale = 0.5
			
			if pred < num_keys:
				if temp < num_keys:
					keyboard.release(keys[temp]) and temp!=pred 
				keyboard.press(keys[pred])
				scale = 1
				message = classes[pred]
			upscaled = cv2.putText(upscaled, message, (4,24), cv2.FONT_HERSHEY_DUPLEX,  
                   scale, (208, 218, 3), 1, cv2.LINE_AA) 

	# display actual camera stream
	frame = imutils.resize(frame, height=250)
	frame = np.concatenate((frame, upscaled), axis=1)
	
	cv2.imshow("Live Feed", imutils.resize(frame, height=250))
	cv2.moveWindow('Live Feed',0,0)

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

