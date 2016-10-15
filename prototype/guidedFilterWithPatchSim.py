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
		dep = cv2.resize(dep, (rgb.shape[0], rgb.shape[1]))
		guide = GuidedFilter(rgb, radius = 2, epsilon = 1e-3)
		out = guide.filter(dep) * 1
		print time.time() - t1
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

		# convert to scalar so we can access array
		for i in range(0, len(pixel1)):
			pixel1[i] = math.floor(pixel1[i])

		return (pixel0, pixel1, st)


	def getImgMask(self, img, pixel, winSize):
		# gets an image mask 
		

		halfWinSize = math.floor(winSize/2)
		sizeW, sizeH = img.shape



		# get each indices to minimize comparisons
		upBound = int(pixel[0] - halfWinSize)
		downBound = int(pixel[0] + halfWinSize)
		leftBound = int(pixel[1] - halfWinSize)
		rightBound = int(pixel[1] + halfWinSize)



		# edge case handling
		if(upBound < 0):
			upBound = 0
		if(downBound >= sizeH):
			downBound = sizeH - 1
		if(leftBound < 0):
			leftBound = 0
		if(rightBound >= sizeW):
			rightBound = sizeW - 1

		# upBound = max(upBound,0)
		# downBound = min(downBound, sizeH - 1)

			# neighborhood slicing
		neighborhood = img[upBound:downBound, leftBound:rightBound]
		return neighborhood		

	def patchSimilarity(self, depImage0, depImage1, sigmaP, winSize):		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel
		# apparently we only need 1 optical flow vector -> but from where?

		# so we use a single patch to compute the total weighting - how to determine patch?
		# use cv2.goodFeaturesToTrack and compute like 4-5 of them to determine weighting
		
		# use single pixel for now, maybe more later?
		
		(img0Feats, img1Feats, st) = self.calculateOptFlow(depImage0, depImage1)


		oldMask = self.getImgMask(depImage0, img0Feats, winSize)
		newMask = self.getImgMask(depImage1, img1Feats, winSize)

		weight = np.exp(-1*np.sum(np.abs(np.subtract(oldMask, newMask))/sigmaP))

		return weight

	def calculateTotalPatchSimilarity(self, prevDepImgs, depImg):
		weights = []
		for i in range(0, len(prevDepImgs)):
			weights.append(self.patchSimilarity(prevDepImgs[i], depImg, 40, 5))

		# just to make sure, let's also calculate the weight for the current image (although)
		# I think it'll be way too high
		# weights.append(self.patchSimilarity(depImg, depImg, 40, 5))
		# print(weights)

		# normalize weights
		for i in range(0, len(prevDepImgs)):
			weights[i] /= sum(weights)

		# now, reconstruct our depth map

		print(weights)
		return weights

	def addNoise(self, depImgs):
		for i in range(0, len(depImgs)):
			noise = np.random.normal(0, 10, (depImgs[0].shape[0], depImgs[0].shape[1])).astype('uint8')
			# depImgs[i] = cv2.resize(depImgs[i], (depImgs[i].shape[0]/4, depImgs[i].shape[1] / 4))
			depImgs[i] += noise

		return depImgs
			

def main():
	imgList = sys.argv[1]
	tf = TemporalFilter(imgList)
	rgb = tf.getRGBImages()
	dep = tf.getDepImages()
	noise = tf.addNoise(dep)
	oldDep = tf.getDepImages()


	# filter with a guide
	for i in range(0, len(dep)):
		dep[i] = tf.filter_with_rgb_guide(rgb[i], noise[i])
		cv2.imshow("depth", dep[i])
		cv2.waitKey(0)


	# weights = tf.calculateTotalPatchSimilarity(dep, dep[-1])



	# outImg = np.zeros(dep[0].shape)
	# for i in range(0, len(weights)):
	# 	outImg += dep[i] * weights[i]

	# cv2.imshow("test", outImg)
	# cv2.waitKey(0)





if __name__ == '__main__':
	main()