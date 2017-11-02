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
f = file("outputFileName.txt", "w")
pathToImage = "<absolutePathToAnImage>"
label = 5
f.write(pathToImage + " " + str(label) + "\n")
f.close()

#################################################################################################
# This is how you would shuffle a list and keep N samples
#################################################################################################
myList = []
myList.append("test1")
myList.append("test2")
myList.append("test3")
myList.append("test4")
myList.append("test5")
myList.append("test6")
numSamplesKeep = 3
reducedRandomList = random.sample(myList, numSamplesKeep)

print myList
print reducedRandomList



