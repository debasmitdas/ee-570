#!/usr/bin/env python
import cv2
import sys
import caffe
import numpy as np

model = "./hw2-deploy.prototxt"
weights = "./hw2-weights.caffemodel"

caffe.set_mode_cpu()

# Set neural network

net = caffe.Net(model, weights, caffe.TEST)
opdict=dict()
opdict[0]="Fox"
opdict[1]="Bike"
opdict[2]="Elephant"
opdict[3]="Car"
inputFile = "./hw2-test-split.txt"
prob=[]
target=[]
correct=0
total=0
nnInputWidth  = 32
nnInputHeight = 32
imgPath=None
with open(inputFile) as f:
	for line in f:
		imgPath=line.split(' ')[0]
		target.append(line.split(' ')[1])
		actual=int(line.split(' ')[1])
		img=cv2.imread(imgPath)
		print(imgPath)
		inputImg = cv2.resize(img, (nnInputWidth, nnInputHeight) , interpolation=cv2.INTER_CUBIC)
		inputImg=inputImg.transpose((2,0,1)) 
		ip=np.zeros((1,3,32,32),dtype='uint8')
		ip[0]=inputImg
		net.blobs['data'].data[...]=ip
		out = net.forward()
		scores = out['prob']
		pred=np.argmax(scores)
		if(actual == pred):
			correct=correct+1
		print("pred Id: "+str(pred)+" - "+opdict[pred])
		print("True Id: "+str(actual)+" - "+opdict[actual])
		prob.append(pred)
		total=total+1

f.close()
target=np.asarray(target)
prob=np.asarray(prob)
acc=correct*100/float(total)
print(correct)
print(total)
print("Accuracy over test Images is: "+str(acc)+"%")