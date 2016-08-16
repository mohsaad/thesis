#!/usr/bin/env python
# Mohammad Saad
# Senior Thesis
# Patch-Sim Reliability Measure

import numpy as np
import cv2


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
	pass


def loadImages(rgbFilename, depthFilename):
	rgb = cv2.imread(rgbFilename)
	print(rgb.shape)
	depth = cv2.imread(depthFilename)
	print(depth.shape)
