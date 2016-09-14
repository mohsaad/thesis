#!/usr/bin/env python
# Mohammad Saad
# 9/6/2016
# guidedFilterWithPatchSim.py
# A guided filter with the patch similarity measure 

import numpy as np
import cv2
from guided_filter.core.filters import FastGuidedFilter, GuidedFilter
import time
import sys
import math

class TemporalFilter:

	def __init__(self, prevFrame, curFrame, nextFrame):
		self.prevFrame = cv2.imread(prevFrame, cv2.IMREAD_GRAYSCALE)
		self.curFrame = cv2.imread(currFrame, cv2.IMREAD_GRAYSCALE)
		self.nextFrame = cv2.imread(nextFrame, cv2.IMREAD_GRAYSCALE)

	def patchSimilarity(depImage0, depImage1, sigmaR, maskRadius):
		# flow = cv2.calcOpticalFlowFarneback(depImage0, depImage1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel

		# first we calculate each optical flow vector to see which pixel went where
		flow = cv2.calcOpticalFlowFarneback(depImage0, depImage1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		print(flow[1][1])

		halfWinSize = math.floor(maskRadius/2)

		# for i in range(0, depImage0.shape[0]): # top/bottom
		# 	for j in range(0, depImage0.shape[1]): # left/right
		# 		# get subsampled image using windowSize

		# 		# get each indices to minimize comparisons
		# 		upBound = i - halfWinSize
		# 		downBound = i + halfWinSize
		# 		leftBound = j - halfWinSize
		# 		rightBound = j + halfWinSize

		# 		# edge case handling
		# 		if(upBound < 0):
		# 			upBound = 0
		# 		if(downBound >= sizeH):
		# 			downBound = sizeH - 1
		# 		if(leftBound < 0):
		# 			leftBound = 0
		# 		if(rightBound >= sizeW):
		# 			rightBound = sizeW - 1

		# 		# neighborhood slicing
		# 		prevNeighborhood = depImage0[upBound:downBound, leftBound:rightBound]
		# 		curreighborhood = depImage1[upBound:downBound, leftBound:rightBound]

		# 		patchSum = np.abs(np.sum(np.subtract(prevNeighborhood, curreighborhood)))



def main():
	prevFrame = sys.argv[1]
	curFrame = sys.argv[]