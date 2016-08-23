#!/usr/bin/env python
# Mohammad Saad
# 8/22/2016
# addDirectoryName.py
# Appends directory names to ordered filenames

import sys

dirName = sys.argv[2]
imageNames = sys.argv[1]
outFileName = sys.argv[3]

f = open(imageNames, 'r')
fw = open(outFileName, 'w')

for line in f:
	fw.write('{0}/{1}\n'.format(dirName, line.split('\n')[0]))

f.close()
fw.close()