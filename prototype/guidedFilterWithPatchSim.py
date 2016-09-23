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
			for i in range(0, int(f.readline())):
				self.rgbImages.append(cv2.imread(f.readline().split("\n")[0]))
			for i in range(0, int(f.readline())):
				self.depImages.append(cv2.imread(f.readline().split("\n")[0], cv2.IMREAD_GRAYSCALE))

		

	def getDepImages(self):
		return self.depImages

	def getRGBImages(self):
		return self.rgbImages

	def filter_with_rgb_guide(self, rgb, dep):
		
		t1 = time.time()
		guide = GuidedFilter(rgb, radius = 1, epsilon = 0.01)
		out = guide.filter(dep)
		return out


	def calculateOptFlow(self, img0, img1):
		feature_params = dict( maxCorners = 100,
                           	qualityLevel = 0.3,
                           	minDistance = 3,
                           	blockSize = 7 )

		testImg0 = np.multiply(img0, 255).astype(np.uint8)
		testImg1 = np.multiply(img1, 255).astype(np.uint8)

   		lk_params = dict(winSize  = (15, 15), maxLevel = 2,
		                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		p0 = cv2.goodFeaturesToTrack(testImg0, mask = None, **feature_params)

		p1, st, err = cv2.calcOpticalFlowPyrLK(testImg0, testImg1, p0, None, **lk_params)

		pixel0 = p0[st == 1][0]
		pixel1 = p1[st == 1][0]

		print(pixel0, pixel1, st)
		return (p0, p1, st)


	def getImgMask(self, img, pixel, winSize):
		# gets an image mask 
		

		halfWinSize = math.floor(maskRadius/2)
		sizeW, sizeH = depImage0.shape

		# get each indices to minimize comparisons
		upBound = pixel[0] - halfWinSize
		downBound = pixel[0] + halfWinSize
		leftBound = pixel[1] - halfWinSize
		rightBound = pixel[1] + halfWinSize

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
		neighborhood = img[upBound:downBound, leftBound:rightBound]
		return neighborhood		

	def patchSimilarity(self, depImage0, depImage1, img0Feats, img1Feats, st, sigmaR, maskRadius):		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel
		# apparently we only need 1 optical flow vector -> but from where?

		# first we calculate each optical flow vector to see which pixel went where
		# this is a nonlocal video denoising - we do a bunch of patches to see where each
		# individual patch went
		# so we use a single patch to compute the total weighting - how to determine patch?
		# use cv2.goodFeaturesToTrack and compute like 4-5 of them to determine weighting
		
		# use single pixel for now, maybe more later?
		# pixel0 = img0Fea

		oldMask = self.getImgMask(depImage0, img0Feats, maskRadius)
		newMask = self.getImgMask(depImage1, img1Feats, maskRadius)


		cv2.imshow('depImage1', depImage1)
		cv2.waitKey(0)
		return depImage1




def main():
	imgList = sys.argv[1]
	tf = TemporalFilter(imgList)
	rgb = tf.getRGBImages()
	dep = tf.getDepImages()
	new_out_1 = tf.filter_with_rgb_guide(rgb[1], dep[1])
	new_out_0 = tf.filter_with_rgb_guide(rgb[0], dep[0])
	tf.calculateOptFlow(new_out_0, new_out_1)


	# tf.patchSimilarity(new_out_0, new_out_1, 5, 40)

if __name__ == '__main__':
	main()