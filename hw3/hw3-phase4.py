#!/usr/bin/env python
#caffe train -solver=/home/min/a/das35/ee570/hw3/hw3-solver.prototxt

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# cv2 is a libary that is useful for image manipulation
# os is a libary that does system type things (like walking directories)
# xml... is a library that lets you parse XML files easily (the annotations are stored in xml files)
# the next three are necessary to import the edge_boxes library that we'll be using
# random helps you sample things in a random fashion
# caffe lets you use caffe
#################################################################################################
import cv2
import os
from os import walk
import sys
sys.path.append('/home/min/a/ee570/edge-boxes-with-python/')
import edge_boxes
import random
import caffe
import numpy as np

#################################################################################################
# This is how you load an image and then grab a "patch" of that image
#################################################################################################

model = "./hw3-deploy.prototxt"
weights = "./hw3-weights.caffemodel"

caffe.set_mode_cpu()

# Set neural network

net = caffe.Net(model, weights, caffe.TEST)
opdict=dict()
opdict[0]="Fox"
opdict[1]="Bike"
opdict[2]="Elephant"
opdict[3]="Car"
opdict[4]="Background"
inputFile = "./hw3-test-split-phase4.txt"
prob=[]
target=[]
correct=0
total=0
nnInputWidth  = 32
nnInputHeight = 32

imgPath=[]
with open(inputFile) as f:
	for line in f:
		imgPath.append(line.split(' ')[0])
		target.append(line.split(' ')[1])
#actual=int(line.split(' ')[1])
#actual=0
windows = edge_boxes.get_windows(imgPath)
for j in range(0,len(windows)):
	img=cv2.imread(imgPath[j])
	actual=int(target[j])
	maxsc=-1; maxid=-1; predmax=-1
	for i in range(0,len(windows[j])):
		prop=[windows[j][i][1], windows[j][i][0], windows[j][i][3], windows[j][i][2]]
		#print(prop[0])
		crop_img=img[int(prop[1]):int(prop[3]), int(prop[0]):int(prop[2])]
		#print(imgPath)
		inputImg = cv2.resize(crop_img, (nnInputWidth, nnInputHeight) , interpolation=cv2.INTER_CUBIC)
		inputImg=inputImg.transpose((2,0,1)) 
		ip=np.zeros((1,3,32,32),dtype='uint8')
		ip[0]=inputImg 
		net.blobs['data'].data[...]=ip
		out = net.forward()
		scores = out['prob']
		#print(scores)
		pred=np.argmax(scores)

		if(pred != 4):
			if(scores[0][pred]>maxsc):
				maxid=i
				maxsc=scores[0][pred]
				predmax=pred
				
	if(actual == predmax):
		correct=correct+1
	print("pred Id: "+str(predmax)+" - "+opdict[predmax])
	print("True Id: "+str(actual)+" - "+opdict[actual])
	prob.append(predmax)
	total=total+1
	cv2.rectangle(img, (int(windows[j][maxid][1]), int(windows[j][maxid][0])), (int(windows[j][maxid][3]), int(windows[j][maxid][2])), (0, 255, 0), 1)
	cv2.imshow(opdict[predmax], img)
	k=cv2.waitKey(3000)
	if(k==32):
		cv2.destroyAllWindows()

#cv2.destroyAllWindows()
#cv2.waitKey(1)



f.close()
target=np.asarray(target)
prob=np.asarray(prob)
acc=correct*100/float(total)
print(correct)
print(total)
print("Accuracy over test Images is: "+str(acc)+"%")

# load an image
#image = cv2.imread('/home/min/a/ee570/hw3-files/hw3-dataset/n02119789/n02119789_14086.JPEG')

# get the patch
#x1 = 50
#y1 = 50
#x2 = 100
#y2 = 150
#proposalImage = image[y1:y2, x1:x2]
#cv2.imshow('origImage', image)
#cv2.imshow('patch', proposalImage)
#print "Press the space bar to exit"
#cv2.waitKey(0)

#################################################################################################
# Everything else you need to do in this file has been provided in previous guides.
#
# All the instructions on how to load caffe, resize an image, transpose the image to match how
# caffe likes the input, put the data in caffe's "data blob", perform a forward pass, extract 
# the predictions, etc. were done in hw2.
#
# All the instructions on how to use edge-boxes is found in hw3-guide-phase1.py
# 
#################################################################################################


