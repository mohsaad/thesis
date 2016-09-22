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

	def __init__(self, imListFile):
		self.rgbImages = []
		self.depImages = []
		with open(imListFile, 'r') as f:
			for i in range(0, int(f.readline()))
				self.rgbImages.append(cv2.imread(f.readline()))
			for i in range(0, int(f.readline()))
				self.depImages.append(cv2.imread(f.readline()))


	def getDepImages(self):
		return self.depImages

	def getRGBImages(self):
		return self.rgbImages

	def filter_with_rgb_guide(self, rgb, dep):
		t1 = time.time()
		guide = GuidedFilter(rgb)
		out = guide.filter(dep)
		cv2.imshow()
		print(time.time() - t1)
		return out


	def patchSimilarity(depImage0, depImage1, depImage2, sigmaR, maskRadius):		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel
		# apparently we only need 1 optical flow vector -> but from where?

		# first we calculate each optical flow vector to see which pixel went where
		# this is a nonlocal video denoising - we do a bunch of patches to see where each
		# individual patch went
		flow = cv2.calcOpticalFlowFarneback(depImage0, depImage1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		print(flow[1][1])

		halfWinSize = math.floor(maskRadius/2)

		for i in range(0, depImage0.shape[0]): # top/bottom
			for j in range(0, depImage0.shape[1]): # left/right
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
				prevNeighborhood = depImage0[upBound:downBound, leftBound:rightBound]
				curreighborhood = depImage1[upBound:downBound, leftBound:rightBound]

				patchSum = np.exp(np.sum(np.abs(np.subtract(prevNeighborhood, curreighborhood))))

				depImage1[i,j] = np.ceil(depImage0[i,j]*patchSum + depImage1[i,j]*(1-patchSum))
				if(depImage1[i,j] > 255):
					depImage1[i,j] = 0





def main():
	imgList = sys.argv[1]

	tf = TemporalFilter(imgList)
	tf


if __name__ == '__main__':
	main()