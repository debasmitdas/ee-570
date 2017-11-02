#!/usr/bin/env python

#################################################################################################
# Import libraries
#################################################################################################
import numpy as np
from numpy import random
np.random.seed() # set random seed for similar results across runs

#################################################################################################
# User-defined variables
#################################################################################################

#specify file names
inputFile       = 'hw3-data.txt'
outputTrainFile = 'hw3-train-split.txt'
outputTestFile  = 'hw3-test-split.txt'

# specify train/val/test ratios
trainRatio = 0.90
testRatio  = 0.10

#################################################################################################
# Count how many items are in the input file
#################################################################################################
numItems = 0

f = open(inputFile,'r')
while True:
	line = f.readline()
	if not line:
		break
	numItems = numItems + 1

print "inputFile: ", inputFile

#################################################################################################
# Shuffle the data randomly
#################################################################################################
nums = np.arange(numItems)
np.random.shuffle(nums)

# compute indeces where things begin/end
trainBeg = 0
trainEnd = int(numItems*trainRatio)
testBeg  = trainEnd
testEnd  = numItems

# extract indeces for each category
trainIndeces = nums[trainBeg:trainEnd];
testIndeces = nums[testBeg:testEnd];

# print numbers
print "\ttotal samples: ", len(trainIndeces) + len(testIndeces)
print "\ttrain samples: ", len(trainIndeces)
print "\ttest samples:  ", len(testIndeces)

#################################################################################################
# Save off the splits into individual files
#################################################################################################
outputTrain = open(outputTrainFile, 'w')
outputTest  = open(outputTestFile, 'w')

# open the orig file
f = open(inputFile,'r')
i = 0;

while True:
	line = f.readline()
	if not line:
		break
	
	if i in trainIndeces:
		outputTrain.write(line)
	
	if i in testIndeces:
		outputTest.write(line)
	
	i = i+1
	
# finish
print('Completed succesfully')
