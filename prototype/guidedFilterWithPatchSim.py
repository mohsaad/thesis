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

class TemporalFilter:

	def __init__(self, filename):
		pass

	def patchSimilarity(depImage0, depImage1):
		# flow = cv2.calcOpticalFlowFarneback(depImage0, depImage1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		
		# so we need to loop over each pixel and compute a mask difference
		# to give us a weighing between each pixel

		# first we calculate each optical flow vector to see which pixel went where
		flow = cv2.calcOpticalFlowFarneback(depImage0, depImage1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		print(flow[1][1])

