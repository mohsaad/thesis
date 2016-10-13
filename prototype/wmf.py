#!/usr/bin/env wmf.py
# Mohammad Saad
# 10/13/2016
# wmf.py
# An implementation of a weighted mode filter for
# comparison to a Guided Mode Filter

# Implementation in Python of the Matlab code here
# https://github.com/qinhongwei/depth-enhancement
# And from the paper Depth video enhancement based on weighted mode filtering by Min et al.

import numpy as np
import cv2


def getImgMask(img, pixel, winSize):
	# gets an image mask 
		
	# in this file, we're using radius, so modify slightly
	halfWinSize = winSize
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

def wmf(rgb, depth, spatial_sigma, color_sigma, depth_sigma, window_radius):

	window = np.arange(-1*window_radius, window_radius, 1)
	[mx, my] = np.meshgrid(window, window)

	spatial_param = np.exp(-1 * (np.pow(window, 2) + np.pow(window, 2)) / (2*spatial_sigma ** 2))

	output = np.zeros(rgb.shape)

	height = rgb.shape[0]
	width = rgb.shape[1]

	for i in range(0, height):
		for j in range(0, width):

			# get the local region
			# same as other script


			rgb_neighborhood = getImgMask(rgb, (i,j), window_radius)
