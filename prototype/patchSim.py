#!/usr/bin/env python
# Mohammad Saad
# Senior Thesis
# Patch-Sim Reliability Measure

import numpy as np
import cv2
import sys
from math import floor
import time
from scipy import ndimage
from scipy.signal import fftconvolve

'''
Taken from here: http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
'''
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
	

'''
Euclidean distance. Used for calculating the spatial Gaussian parameter.
'''
def distance(p1, p2):
	return 0

'''
Assuming RGB and depth images are the same size, calculate the joint histogram
'''
def generate_joint_histogram(rgbImage, depImage, windowSize, sigmaR, sigmaS, sigmaI, nBins):
	sizeH = rgbImage.shape[0]
	sizeW = rgbImage.shape[1]
	t0 = time.time()
	histogram = np.zeros([sizeH, sizeW, nBins])
	halfWinSize = int(floor(windowSize / 2.0))

	t1 = time.time()
	spatialFilter = matlab_style_gauss2D(shape = (windowSize, windowSize), sigma = sigmaS)
	guideFilter = matlab_style_gauss2D(shape = (windowSize, windowSize), sigma = sigmaI)
	relaxationFilter = matlab_style_gauss2D(shape = (windowSize, windowSize), sigma = sigmaR)
	print('Initialization time: {0} s'.format(time.time() - t0))
	# compute distances 


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



			# numpy-ized
			colorDiff = np.sum(np.abs(np.subtract(rgbImage[i,j,:], rgbNeighborhood[:,:,:])), axis = 2)
			guideFiltered = fftconvolve(guideFilter, colorDiff)







	t1 = time.time()
	print(t1 - t0)


'''
Calculates optical flow. Images should be ordered.
'''
def calcOpticalFlowVectors(depImage1, depImage2):
	t0 = time.time()
	flow = cv2.calcOpticalFlowFarneback(depImage1, depImage2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	t1 = time.time()
	print(t1 - t0)
	print(flow.shape)

'''
Loads a bunch of depth images
'''
def loadDepImages(depFileName, numImages):
	imgs = []
	count = 0
	with open(depFileName, 'r') as f:
		for line in f:
			print(count)
			if count > numImages:
				break
			imgs.append(cv2.imread(line.split('\n')[0], cv2.IMREAD_GRAYSCALE))
			count += 1
	return imgs

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
	generate_joint_histogram(rgb, dep, 5, 1,1,1,256)

	# depImgs = loadDepImages('testOptFlow.txt', 3)

	# calcOpticalFlowVectors(depImgs[0], depImgs[1])



