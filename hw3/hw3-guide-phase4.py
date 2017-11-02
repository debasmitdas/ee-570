#!/usr/bin/env python

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

#################################################################################################
# This is how you load an image and then grab a "patch" of that image
#################################################################################################

# load an image
image = cv2.imread('/home/min/a/ee570/hw3-files/hw3-dataset/n02119789/n02119789_14086.JPEG')

# get the patch
x1 = 50
y1 = 50
x2 = 100
y2 = 150
proposalImage = image[y1:y2, x1:x2]
cv2.imshow('origImage', image)
cv2.imshow('patch', proposalImage)
print "Press the space bar to exit"
cv2.waitKey(0)

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


