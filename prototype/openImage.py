#!/usr/bin/env python
# Mohammad Saad
# 8/12/2016
# Senior Thesis
# Takes a folder of RGB + depth images and converts them into PNG files

from PIL import Image
import sys
from os import listdir
from os.path import isfile, join, isdir
from os import mkdir, rmdir

class ImageConverter:

	def __init__(self, pathname):
		self.pathname = pathname

	def getFilesInPath(self):

		self.filenames = [f for f in listdir(self.pathname) if isfile(join(self.pathname, f))]
		for i in range(0, len(self.filenames)):
			self.filenames[i] = '{0}/{1}'.format(self.pathname, self.filenames[i])

	def writeFilesAsImages(self, dirname):
		if(isdir(dirname)):
			rmdir(dirname)

		mkdir(dirname)
		countRGB = 0
		countD = 0
		for i  in range(0, len(self.filenames)):
			file = self.filenames[i]
			fileArr = file.split('.')
			if(fileArr[-1] == 'ppm'):
				rgb = Image.open(file)
				rgb.save('{0}/rgb{1}.png'.format(dirname,str(countRGB)))
				rgb.close()
				countRGB += 1
			elif(fileArr[-1] == 'pgm'):
				depth = Image.open(file)
				depth.save('{0}/d{1}.png'.format(dirname, str(countD)))
				depth.close()
				countD += 1 
			print '{0},{1}'.format(str(countRGB), str(countD))
		print "Conversion complete!"

	def writeDepthVideoSequence(self, dirname, depthOrderFile):
		countD = 0
		if not isdir(dirname):
			mkdir(dirname)
		elif not isdir(dirname+'/depth'):
			mkdir(dirname+'/depth')

		with open(depthOrderFile, 'r') as f:
			for line in f:
				line = line.split('\n')[0]
				depth = Image.open(self.pathname + line)
				depth.save('{0}/depth/d{1}.png'.format(dirname, str(countD)))
				depth.close()
				countD += 1 
				print('Writing image ' + str(countD))

		f.close()


	def writeRGBVideoSequence(self, dirname, depthOrderFile):
		countD = 0
		if not isdir(dirname):
			mkdir(dirname)
		elif not isdir(dirname+'/rgb'):
			mkdir(dirname+'/rgb')

		with open(depthOrderFile, 'r') as f:
			for line in f:
				line = line.split('\n')[0]
				depth = Image.open(self.pathname + line)
				depth.save('{0}/rgb/rgb{1}.png'.format(dirname, str(countD)))
				depth.close()
				countD += 1 
				print('Writing image ' + str(countD))

		f.close()

if __name__ == '__main__':
	ic = ImageConverter(sys.argv[1])
	# ic.writeDepthVideoSequence(sys.argv[2], sys.argv[3])
	ic.writeRGBVideoSequence(sys.argv[2], sys.argv[3])
