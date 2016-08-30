#!/usr/bin/env python
# Mohammad Saad
# 8/25/2016
# tests a guided filter operation

import numpy as np
import cv2
from guided_filter.core.filters import FastGuidedFilter, GuidedFilter
import sys
import time

'''
Calculates optical flow. Images should be ordered.
'''
def calcOpticalFlowVectors(depImage1, depImage2):
	t0 = time.time()
	flow = cv2.calcOpticalFlowFarneback(depImage1, depImage2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	t1 = time.time()
	return flow

def filterWithGuide(depImage0, rgbImage0, sigma, epsilon):
	guidedFilt = FastGuidedFilter(dep0, radius = sigma, epsilon = epsilon)
	


count = 0
with open(sys.argv[1], 'r') as f:
	for line in f:
		filename = '/home/saad/Code/thesis/test/{0}'.format(line.split('\n')[0])
		print(filename)
		dep0 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		sigma = 0.05

		t0 = time.time()

		guidedFilt = FastGuidedFilter(dep0, radius = sigma, epsilon = 0.02)
		C_smooth = guidedFilt.filter(dep0)
		if(count != 0):
			calcOpticalFlowVectors(prevDep0, C_smooth)

		print(time.time() - t0)
	
		cv2.imshow('filtered', C_smooth)
		cv2.waitKey(1000)
		prevDep0 = C_smooth
		count += 1
f.close()

# dep0 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# sigma = 1
# guidedFilt = FastGuidedFilter(dep0, radius = sigma, epsilon = 0.02)
# C_smooth = guidedFilt.filter(dep0)

# cv2.imshow('original', dep0)
# cv2.waitKey(0)

# cv2.imshow('filtered', C_smooth)
# cv2.waitKey(0)

# dep1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

# sigma = 1
# guidedFilt = FastGuidedFilter(dep0, radius = sigma, epsilon = 0.02)
# C_smooth = guidedFilt.filter(dep0)

# cv2.imshow('original2', dep0)
# cv2.waitKey(0)

# cv2.imshow('filtered2', C_smooth)
# cv2.waitKey(0)
