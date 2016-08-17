#!/usr/bin/env python
# Mohammad Saad
# Senior Thesis
# Patch-Sim Reliability Measure

import numpy as np
import cv2
import sys
from math import floor
import time

def gaussianI(sigma, value):
	pass

'''
Euclidean distance. Used for calculating the spatial Gaussian parameter.
'''
def distance(p1, p2):
	pass

'''
Assuming RGB and depth images are the same size, calculate the joint histogram
'''
def generate_joint_histogram(rgbImage, depImage, windowSize, sigmaR, sigmaS, sigmaI, nBins):
	sizeH = rgbImage.shape[0]
	sizeW = rgbImage.shape[1]
	t0 = time.time()
	histogram = np.zeros([sizeH, sizeW, nBins])
	halfWinSize = int(floor(windowSize / 2.0))

	# for each pixel, generate a 256-bin histogram
	for i in range(0, depImage.shape[0]): # top/bottom
		for j in range(0, rgbImage.shape[1]): # left/right
			# get subsampled image using windowSize

			# get each indices to minimize comparisons
			upBound = i - halfWinSize
			downBound = i + halfWinSize
			leftBound = j - halfWinSize
			rightBound = j + halfWinSize

			# edge case handling
			if(upBound < 0):
				upBound = 0
			if(downBound >= sizeH):
				downBound = sizeH - 1
			if(leftBound < 0):
				leftBound = 0
			if(rightBound >= sizeW):
				rightBound = sizeW - 1

			# neighborhood slicing
			depNeighborhood = depImage[upBound:downBound, leftBound:rightBound]
			rgbNeighborhood = rgbImage[upBound:downBound, leftBound:rightBound, :]

			# loop over each neighborhood pixel, put in bin
			# there has got to be a better way to do this
			for k in range(0, nBins):
				for m in range(0, depNeighborhood.shape[0]):
					for n in range(0, depNeighborhood.shape[1]):
						colorDiff = sum(abs(rgbImage[i,j,:] - rgbNeighborhood[m,n,:]))
						histogram[i,j,k] += (gaussian(sigmaS, distance([i,j], [m,n])) * 
										gaussian(sigmaR, k - depNeighborhood[m,n]) * 
										gaussian(sigmaI, colorDiff)




	t1 = time.time()
	print(t1 - t0)
def loadImages(rgbFilename, depthFilename):
	rgb = cv2.imread(rgbFilename)
	depth = cv2.imread(depthFilename, cv2.IMREAD_GRAYSCALE)
	# compare shapes, convert RGB to CIELAB space
	if(rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]):
		print("Resolutions are not equal. Please correct.")
		sys.exit()


	#cv2.imshow("test",rgb[:,:,:])
	#cv2.waitKey(0)
	# convert to CIELAB space
	# lab = cv2.cvtColor(rgb, cv2.CV_BGR2Lab)
	return (rgb, depth)

if __name__ == '__main__':
	(rgb, dep) = loadImages(sys.argv[1], sys.argv[2])
	generate_joint_histogram(rgb, dep, 7, 0,0,0,256)