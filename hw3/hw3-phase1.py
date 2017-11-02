#!/usr/bin/env python

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# cv2 is a libary that is useful for image manipulation
# os is a libary that does system type things (like creating directories)
# xml... is a library that lets you parse XML files easily (the annotations are stored in xml files)
# the last three are necessary to import the edge_boxes library that we'll be using
#################################################################################################
import cv2	
import os
from os import walk
import xml.etree.ElementTree as ET
import sys
sys.path.append('/home/min/a/ee570/edge-boxes-with-python/')	
import edge_boxes
import time

#################################################################################################
# User-defined variables
#################################################################################################
# This is where the dataset that we'll be working with lives.  
# Since you will be parsing the directory structure, it'd probably be useful to go there and
# see what it looks like.  Type "nautilus /home/min/a/ee570/hw3-files/hw3-dataset &" to open a
# file browser in that directory so you can poke around.
datasetDir   = '/home/min/a/ee570/hw3-files/hw3-dataset'

# The classes, their ID's, and labels are as follows:
# 	CLASS NAME		CLASS ID		LABEL
#	fox				n02119789		0
#	bike			n03792782		1
#	elephant		n02504458		2
#	car				n04037443		3

#################################################################################################
# Function to compute IoU
# Code lifted from here (THANKS!): 
#	http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# Added check at beginning to determine if the boxes actually overlap
# function takes in two variables, both of which are 1x4 arrays that look like this:
# box[0] = topLeftX coordinate
# box[1] = topLeftY coordinate
# box[2] = bottomRightX coordinate
# box[3] = bottomRightY coordinate
#################################################################################################
def bb_intersection_over_union(boxA, boxB):
	# determine if the boxes don't overlap at all
	if (boxB[0] > boxA[2]) or (boxA[0] > boxB[2]) or (boxB[1] > boxA[3]) or (boxA[1] > boxB[3]):
		return 0

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


#################################################################################################
# This is how you would get a list of all files in a directory
#################################################################################################

listcind=["/n02119789","/n03792782","/n02504458","/n04037443"]
listnames=["fox","bike","elephant","car"]

for clind in range(0,4):
	imageDir = datasetDir + listcind[clind];
	imageFiles = []

	for (dirpath, dirnames, filenames) in walk(imageDir):
		imageFiles.extend(filenames)
		break

	#print "Printing list of image files"
	#print imageFiles
	#print "num Image Files in n02119789: ", len(imageFiles)


	#################################################################################################
	# This is how you would get the edge boxes for an image
	#
	# imageFileNames is a python list with absolute paths to images (the list can be of length 1)
	#
	# windows is a python list that looks like this:
	# 		windows[imageIdx][boxIdx][0] = topLeftY
	# 		windows[imageIdx][boxIdx][1] = topLeftX
	# 		windows[imageIdx][boxIdx][2] = bottomRightY
	# 		windows[imageIdx][boxIdx][3] = bottomRightX
	# 		windows[imageIdx][boxIdx][4] = quality of proposal (higher number = more likely to be an object)
	#
	# NOTE: imageIdx has N elements, where N is the number of images you passed into the function
	#		N = 2 in the example below	
	# NOTE: the size of boxIdx varies for each image.  In other words, on image1, there might be 2318 
	#		boxes...and on image2, there might be 809 boxes.
	# NOTE: When you are developing your code, I would recommend passing in a single image to the 
	#		function below.  It took ~15 seconds on ecegrid to compute proposals on a single image and
	#		return the results.
	#		After your code is working properly, and you are ready to run the algorithm on all the images
	#		in the directory (ideally you have a nice loop setup to do this), I would recommend you tweak
	#		your code so you only call edge_boxes once (and pass in a list with paths to all the images).
	#		Why?  Of the 15 seconds, something like 14.5 of them are "waiting for Matlab to load".  Only
	#		0.5 of them are doing work.  If you have to loop over 400 images, waiting 14.5 seconds per image
	#		(for Matlab to load) adds up to something like 100 minutes.  If you call the edge_boxes function
	#		once (with paths to all 400 images), it only takes 3 minutes to run on ecegrid.
	# 		SO... again my advice: develop your code by passing in a single image (since waiting 15 seconds
	#		is better than waiting 3 minutes).  Then, when you're ready to run your code on the entire 
	#		directory of images, tweak your code to call edge_boxes once.
	# NOTE:	When I had finished coding and was ready to run my code on the entire directory of images,
	#		it took ~15 minutes from beginning to end.  Each directory of images (4 total) took ~3 minutes
	#		for edge_boxes to run and ~1.5 minutes for the discovery, sorting, and saving of pos/neg images.  
	#		What can you do while you wait 15 minutes for your code to complete?  Start working on phase2!
	#################################################################################################
	imageFileNames = []
	for fi in range(0,len(imageFiles)):
		imageFileNames.append(imageDir + "/" + imageFiles[fi])
		
	windows = edge_boxes.get_windows(imageFileNames)
	pt=0;nt=0;
	t0=time.time()
	for fi in range(0,len(imageFiles)):
		annotationFileName = datasetDir + "/annotation"+listcind[clind]+"/"+imageFiles[fi][:-5]+".xml"

		# load the annotation file into a data structure
		tree = ET.parse(annotationFileName)
		objs = tree.findall('object') # "object" is the tag for an instance of the class.  There can be multiple
								  # "object" tags if there are multiple instances of the class in the image.
		#numObjs = len(objs) # this is how many ground truth instances there are 

		# This is one way of looping over the instances and extracting their coordinates
		groundTruthBox=[]
		for ix, obj in enumerate(objs):
			bbox = obj.find('bndbox') # bndbox is a tag inside of "object".  It has the ground truth coordinates
			groundTruth_x1 = int(bbox.find('xmin').text)
			groundTruth_y1 = int(bbox.find('ymin').text)
			groundTruth_x2 = int(bbox.find('xmax').text)
			groundTruth_y2 = int(bbox.find('ymax').text)

			# here I put the four coordinates into a "box" that I might pass into my bb_intersection_over_union function.  (#useful!)
			groundTruthBox.append([groundTruth_x1, groundTruth_y1, groundTruth_x2, groundTruth_y2])
		  
		pcnt=0;ncnt=0;
		prpfig=windows[fi]
		for bi in range(0,len(windows[fi])):
		
			prop=[prpfig[bi][1], prpfig[bi][0], prpfig[bi][3], prpfig[bi][2]]
			#################################################################################################
			# This is how you get the annotations for an image
			# It would behoove you to go look at one of the XML files so you know what they look like.
			# The code below will make more sense if you do.
			#################################################################################################
			maxiou=0;
			for itera in range(0,len(groundTruthBox)):
				iou = bb_intersection_over_union(groundTruthBox[itera], prop)
				if(iou>maxiou):
					maxiou=iou
			if (maxiou>0.7 and pcnt<5):
				#True proposal
				pcnt=pcnt+1
				pt=pt+1
				img=cv2.imread(imageFileNames[fi])
				crop_img=img[int(prop[1]):int(prop[3]), int(prop[0]):int(prop[2])]
				#crop_img=img[int(windows[fi][bi][0]):int(windows[fi][bi][2]), int(windows[fi][bi][1]):int(windows[fi][bi][3])]
				string="images/"+listnames[clind]+"/pos/image_"+str(pt+10000)[1:]+".jpeg"
				cv2.imwrite(string, crop_img)
			elif (maxiou<0.4 and ncnt<5):
				#False proposal
				ncnt=ncnt+1
				nt=nt+1
				img=cv2.imread(imageFileNames[fi])
				crop_img=img[int(prop[1]):int(prop[3]), int(prop[0]):int(prop[2])]
				#crop_img=img[int(windows[fi][bi][0]):int(windows[fi][bi][2]), int(windows[fi][bi][1]):int(windows[fi][bi][3])]
				string="images/"+listnames[clind]+"/neg/image_"+str(nt+10000)[1:]+".jpeg"
				cv2.imwrite(string, crop_img)
			elif (pcnt==5 and ncnt==5):
				break

	t1=time.time()

	print(t1-t0)
	


#################################################################################################
# This is how you would check if a directory exists...and if it doesn't, create it
#################################################################################################
#if not os.path.exists("images"):
#	os.makedirs("images")


#################################################################################################
# Here we put a lot of the pieces together to load an image and display a bounding box around
# the instance of the class.
#################################################################################################

# load the image

#image = cv2.imread(datasetDir + "/n02119789/n02119789_14086.JPEG")

# load the annotations
#tree = ET.parse(datasetDir + "/annotation/n02119789/n02119789_14086.xml")
#objs = tree.findall('object')

# This is one way of looping over the instances and extracting their coordinates
#for ix, obj in enumerate(objs):
#	bbox = obj.find('bndbox') # bndbox is a tag inside of "object".  It has the ground truth coordinates
#	x1 = int(bbox.find('xmin').text)
#	y1 = int(bbox.find('ymin').text)
#	x2 = int(bbox.find('xmax').text)
#	y2 = int(bbox.find('ymax').text)
	
	# draw bounding boxes on the image
#	cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

#cv2.imshow('Cute foxes', image)
#print "Wasn't this a cute demo?"
#print "Press the space bar to close the image of the foxes."
#cv2.waitKey(0)

