#!/usr/bin/env python

#################################################################################################
# Import libraries
#################################################################################################
import cv2
import sys
import caffe

#################################################################################################
# THIS IS HOW I PRINT THINGS
#################################################################################################
print("\nHello World!  I am going to learn caffe!\n\n")
print("Now I will spit out some errors.")

#################################################################################################
# THIS IS HOW I LOAD A CAFFE MODEL
#################################################################################################
model = "path/to/deploy.prototxt"
weights = "path/to/_iter###.caffemodel"

# specify caffe mode of operation
caffe.set_mode_cpu()

# create net object
net = caffe.Net(model, weights, caffe.TEST)

#################################################################################################
# THIS IS HOW I READ A TEXT FILE LINE BY LINE
#################################################################################################
inputFile = "path/to/inputFile.txt"

f = open(inputFile,'r')
while True:
	line = f.readline()
	if not line:
		break
	print(line)

#################################################################################################
# THIS IS HOW I LOAD AN IMAGE AND SHOW IT
#################################################################################################
imagePath = "path/to/image.JPEG"

# load image
image = cv2.imread(imagePath)

# show image
cv2.imshow('image', image)
cv2.waitKey(0)

#################################################################################################
# THIS IS HOW I FEED AN IMAGE INTO MY NN, DO A FORWARD PASS, AND GET THE OUTPUT
#################################################################################################
nnInputWidth  = 96
nnInputHeight = 96

# resize image to NN input dimensions
inputResImg = cv2.resize(image, (nnInputWidth, nnInputHeight) , interpolation=cv2.INTER_CUBIC)

# this is to rearrange the data to match how caffe likes the input
transposedInputImg = inputResImg.transpose((2,0,1))

# put image into data blob
net.blobs['data'].data[...]=transposedInputImg

# do a forward pass
out = net.forward()

# get the predicted scores
scores = out['loss']

