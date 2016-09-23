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
		print(time.time() - t1)
		return out


	def calculateOptFlow(self, img0, img1, winSize, maxLevel):
		feature_params = dict( maxCorners = 100,
                           	qualityLevel = 0.3,
                           	minDistance = 7,
                           	blockSize = 7 )
   
		 lk_params = dict( winSize  = (15,15), maxLevel = 2,
		                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		 p0 = cv2.goodFeaturesToTrack(img0, mask = None, **feature_params)


	def patchSimilarity(self, depImage0, depImage1, sigmaR, maskRadius):		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel
		# apparently we only need 1 optical flow vector -> but from where?

		# first we calculate each optical flow vector to see which pixel went where
		# this is a nonlocal video denoising - we do a bunch of patches to see where each
		# individual patch went
		# so we use a single patch to compute the total weighting - how to determine patch?
		# use cv2.goodFeaturesToTrack and compute like 4-5 of them to determine weighting
		
		print(flow[1][1])

		halfWinSize = math.floor(maskRadius/2)
		sizeW, sizeH = depImage0.shape




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
	print(new_out_0.shape, new_out_1.shape)

	tf.patchSimilarity(new_out_0, new_out_1, 5, 40)

if __name__ == '__main__':
	main()