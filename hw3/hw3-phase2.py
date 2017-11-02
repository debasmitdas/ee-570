#!/usr/bin/env python

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# os is a libary that does system type things (like walking directories)
# random helps you sample things in a random fashion
#################################################################################################
import os
from os import walk
import random

#################################################################################################
# This is how you would open a file, write to the file, and close the file
#################################################################################################
datasetDir   = '/home/min/a/das35/ee570/hw3/images/'
listnames=["fox","bike","elephant","car"]
f = file("hw3-data.txt", "w")

for clind in range(0,4):
	imageDir = datasetDir + listnames[clind]+"/pos/";
	imageFiles = [];
	for (dirpath, dirnames, filenames) in walk(imageDir):
		imageFiles.extend(filenames)
	redimgFiles = random.sample(imageFiles, 600)
	for i in range(0,len(redimgFiles)):
		pathToImage = imageDir + redimgFiles[i]
		f.write(pathToImage + " " + str(clind) + "\n")
	
	imageDir = datasetDir + listnames[clind]+"/neg/";
	imageFiles = [];
	for (dirpath, dirnames, filenames) in walk(imageDir):
		imageFiles.extend(filenames)
	redimgFiles = random.sample(imageFiles, 150)
	for i in range(0,len(redimgFiles)):
		pathToImage = imageDir + redimgFiles[i]
		f.write(pathToImage + " " + str(4) + "\n")
		
f.close()

#################################################################################################
# This is how you would shuffle a list and keep N samples
#################################################################################################
#myList = []
#myList.append("test1")
#myList.append("test2")
#myList.append("test3")
#myList.append("test4")
#myList.append("test5")
#myList.append("test6")
#numSamplesKeep = 3
#reducedRandomList = random.sample(myList, numSamplesKeep)

#print myList
#print reducedRandomList



