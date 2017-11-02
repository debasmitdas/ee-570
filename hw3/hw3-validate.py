#!/usr/bin/env python
#################################################################################################
# 	Filename: hw2-validate.py
#	Author:   Jared Johansen
#################################################################################################

#################################################################################################
# Import libraries
#################################################################################################
import cv2
import sys
import caffe
import os.path

#################################################################################################
# Grader-script, hard-coded values (DO NOT CHANGE THESE)
#################################################################################################
model     = 'hw3-deploy.prototxt';
weights   = 'hw3-weights.caffemodel';
imagePath = '/home/min/a/ee570/hw3-files/hw3-dataset/n04037443/n04037443_17929.JPEG'
nnInputHeight = 32
nnInputWidth  = 32

#################################################################################################
# Print instructions
#################################################################################################
print("\nThis script will check several if your NN meets basic submission criteria.\n")

#################################################################################################
# Check if files are named correctly
#################################################################################################
if (os.path.isfile(model)) and (os.path.isfile(weights)):
	print("\tFiles named correctly:            SUCCESS")
else:
	print("\tFiles named correctly:            FAILURE")
	print("\tYou must have a file named '" + model + "' and '" + weights + "'.")
	sys.exit()

#################################################################################################
# Check if caffe model and weights load correctly
#################################################################################################
# specify caffe mode of operation
caffe.set_mode_cpu()

# create net
try:
	net = caffe.Net(model, weights, caffe.TEST) 
	# if this crashes, it does a core dump.  There is no graceful way I've found to catch that in python...
	# TODO: Find some graceful way to print to the use the this failed
except: 
	print("\tAble to load your file:           FAILURE")
	sys.exit()

#################################################################################################
# Reprint stuff in case the caffe network loaded and spit out a bunch of stuff to the console
#################################################################################################
print("\nThis script will check several if your NN meets basic submission criteria.\n")
print("\tFiles named correctly:            SUCCESS")
print("\tAble to load your file:           SUCCESS")

#################################################################################################
# Check if input blob is named correctly
#################################################################################################
# load image, resize to NN input dimensions, transpose to fit caffe's input preference
image = cv2.imread(imagePath)
inputResImg = cv2.resize(image, (nnInputWidth, nnInputHeight) , interpolation=cv2.INTER_CUBIC)
transposedInputImg = inputResImg.transpose((2,0,1))

# put image into data blob
try:
	net.blobs['data'].data[...]=transposedInputImg
	print("\tInput layer is named correctly:   SUCCESS")
except:
	print("\tInput layer is named correctly:   FAILURE")
	print("\tYour input layer should be named 'data'.  (It should have been trained this way too.)")
	sys.exit()

#################################################################################################
# Check if network can do a forward pass
#################################################################################################
try:
	out = net.forward()
	print("\tNetwork can do a forward pass:    SUCCESS")
except:
	print("\tNetwork can do a forward pass:    FAILURE")
	sys.exit()

#################################################################################################
# Check if output blob is named correctly
#################################################################################################
try:
	scores = out['prob']
	print("\tOutput layer is named correctly:  SUCCESS")
except:
	print("\tOutput layer is named correctly:  FAILURE")
	print("\tThe top blob of your output layer should be named 'prob'.")
	sys.exit()

#################################################################################################
# Print submission instructions
#################################################################################################
print("\nSUBMISSION INSTRUCTIONS:")
print("\tPut the following files into a folder called 'hw3':")
print("\t\t'hw3-train.prototxt'")
print("\t\t'hw3-solver.prototxt'")
print("\t\t'hw3-deploy.prototxt'")
print("\t\t'hw3-weights.caffemodel'")
print("\t\t'hw3-phase1.py'")
print("\t\t'hw3-phase2.py'")
print("\t\t'hw3-phase4.py'")
print("\tZIP the 'hw3' folder.  (No tarballs, please.)")
print("\tUpload the ZIPPED folder to Blackboard.")
print("\tDo not upload anything else.\n\n")

