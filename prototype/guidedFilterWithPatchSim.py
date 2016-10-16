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
		dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]))
		guide = GuidedFilter(rgb, radius = 2, epsilon = 1e-3)
		out = guide.filter(dep) * 1
		return out


	def calculateOptFlow(self, img0, img1):
		feature_params = dict( maxCorners = 100,
                           	qualityLevel = 0.3,
                           	minDistance = 3,
                           	blockSize = 7 )

		testImg0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
		testImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

		# testImg0 = np.multiply(img0, 255).astype(np.uint8)
		# testImg1 = np.multiply(img1, 255).astype(np.uint8)

   		lk_params = dict(winSize  = (15, 15), maxLevel = 2,
		                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		p0 = cv2.goodFeaturesToTrack(testImg0, mask = None, **feature_params)


		p1, st, err = cv2.calcOpticalFlowPyrLK(testImg0, testImg1, p0, None, **lk_params)



		pixel0 = p0[st == 1][1]
		pixel1 = p1[st == 1][1]

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

	def patchSimilarity(self, depImage0, depImage1, rgb0, rgb1, sigmaP, winSize):		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel
		# apparently we only need 1 optical flow vector -> but from where?

		# so we use a single patch to compute the total weighting - how to determine patch?
		# use cv2.goodFeaturesToTrack and compute like 4-5 of them to determine weighting
		
		# use single pixel for now, maybe more later?
		
		(img0Feats, img1Feats, st) = self.calculateOptFlow(rgb0, rgb1)


		oldMask = self.getImgMask(depImage0, img0Feats, winSize)
		newMask = self.getImgMask(depImage1, img1Feats, winSize)

		if(oldMask.shape != newMask.shape):
			return 0

		weight = np.exp(-1*np.sum(np.abs(np.subtract(oldMask, newMask))/sigmaP))

		return weight

	def calculateTotalPatchSimilarity(self, dep, rgb, sigmaP, winSize):
		weights = []
		out_dep = []

		# so we're going to use the method implemented in Minh's paper
		# Which is to calculate the weight for t-1 frame and the t+1 frame
		# first one has no improvement
		out_dep.append(dep[0])
		for i in range(1, len(dep) - 1):
			# get weight of t-1 frame
			prevWeight = self.patchSimilarity(dep[i-1],dep[i],rgb[i-1],rgb[i], sigmaP, winSize)
			# get weight of t+1 frame
			nextWeight = self.patchSimilarity(dep[i],dep[i+1],rgb[i],rgb[i+1], sigmaP, winSize)

			# get weight of current frame
			currWeight = self.patchSimilarity(dep[i], dep[i], rgb[i], rgb[i], sigmaP, winSize)

			total_weight = [prevWeight, currWeight, nextWeight]

			for j in range(0, len(total_weight)):
				total_weight[j] /= float(sum(total_weight))


			# in the paper, the current frame has a weight equivalent to the 
			dep[i] = (0.25 * dep[i-i] + 0.5 * dep[i] + 0.25 * dep[i + 1])

		# out_dep[len(dep) - 1].append(dep[-1])
		return dep

	def addNoise(self, depImgs):
		for i in range(0, len(depImgs)):
			noise = np.random.normal(0, 10, (depImgs[0].shape[0], depImgs[0].shape[1])).astype('uint8')
			# depImgs[i] = cv2.resize(depImgs[i], (depImgs[i].shape[0]/4, depImgs[i].shape[1] / 4))
			depImgs[i] += noise
			cv2.imwrite("../noise_imgs/depNoise{0}.png".format(i), depImgs[i])

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
		# cv2.imshow("depth", dep[i])
		# cv2.waitKey(0)


	patchSim = tf.calculateTotalPatchSimilarity(dep, rgb, 40, 4)

	for i in range(0, len(patchSim)):
		cv2.imshow("test", patchSim[i])
		cv2.waitKey(0)





if __name__ == '__main__':
	main()