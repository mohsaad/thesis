#!/usr/bin/env python
# Mohammad Saad
# Senior Thesis
# Patch-Sim Reliability Measure

import numpy as np
import cv2
import sys
from math import floor
import time

def gaussian(sigma, value):
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
	t0 = time.time()
	histogram = np.zeros(depImage.shape[0], depImage.shape[1], nBins)
	halfWinSize = floor(windowSize / 2.0)
	sizeH = rgbImage.shape[0]
	sizeW = rgbImage.shape[1]
	for i in range(0, depImage.shape[0]): # top/bottom
		for j in range(0, rgbImage.shape[1]): # left/right
			# get subsampled image using windowSize

			# get each indices to minimize comparisons
			upBound = i - halfWinSize
			downBound = i + halfWinSize
			leftBound = j - halfWinSize
			rightBound = j + halfWinSize

			# mask is a tuple for array slicing
			mask = ([upBound, downBound],[leftBound, rightBound])
			if(upBound < 0):
				mask[0][0] = 0
			if(downBound >= sizeH):
				mask[0][1] = sizeH - 1
			if(leftBound < 0):
				mask[1][0] = 0
			if(rightBound >= sizeW):
				mask[1][1] = sizeW - 1

			rgbNeighborhood = rgbImage[mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]]
			depNeighborhood = depImage[mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]]


	t1 = time.time()
	print(t1 - t0)
def loadImages(rgbFilename, depthFilename):
	rgb = cv2.imread(rgbFilename)
	depth = cv2.imread(depthFilename, cv2.IMREAD_GRAYSCALE)
	# compare shapes, convert RGB to CIELAB space
	if(rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]):
		print("Resolutions are not equal. Please correct.")
		sys.exit()


	cv2.imshow("test",rgb[:,:,:])
	cv2.waitKey(0)
	# convert to CIELAB space
	lab = cv2.cvtColor(rgb, cv2.CV_BGR2Lab)


if __name__ == '__main__':
	loadImages(sys.argv[1], sys.argv[2])