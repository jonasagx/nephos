import sys
import csv

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from base64 import b64decode
from pymongo import DESCENDING
from pymongo import MongoClient
from scipy.spatial import distance

class Detector:
	def __init__(self):
		pass

	def detectAndCompute(self, im, noneValue):
		pass

class KeyPoint:
	def __init__(self, x, y, size):
		self.pt = [x, y]
		self.size = size

class NegriDetector(Detector):
	def __init__(self, min, max, featureXsize, featureYsize):
		# grayscale threshold
		self.min = min
		self.max = max

		# feature window dimensions
		if not (featureXsize % 2 == 0 and featureYsize % 2 == 0):
			raise Exception("Dimensions should both be even")

		self.featureXsize = featureYsize
		self.featureYsize = featureYsize

	def prepareImage(self, im):
		grey = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
		ret, th1 = cv.threshold(grey, self.min, self.max, cv.THRESH_BINARY)
		return th1

	# Retuns keypoints and descriptor
	def detectAndCompute(self, im, noneValue):
		grey = self.prepareImage(im)
		mx, my = grey.shape
		imageArea = mx * my
		featureArea = self.featureXsize * self.featureYsize

		if (imageArea < featureArea):
		  raise Exception("The feature is too big to fit in the image", imageArea, featureArea)

		keypoints, descriptors = [], []
		X, Y = grey.shape
		
		for i in range(X):
			for j in range(Y):
				if grey.item(i, j) == 255:
					feature = self.takeFeature(grey, i, j)
					descriptors.append(feature)
					keypoints.append(KeyPoint(i, j, self.min))

		return (keypoints, descriptors)

	def takeFeature(self, im, i, j):
		xw0, yw0, xw1, yw1 = self.checkFeatureWindow(i, j, im)

		feature = im[xw0:xw1, yw0:yw1]
		im[xw0:xw1, yw0:yw1] = 0
		return feature

	def checkFeatureWindow(self, i, j, im):
		mx, my = im.shape

		xw0 = i - self.featureXsize/2
		yw0 = j - self.featureYsize/2
		
		xw1 = i + self.featureXsize/2
		yw1 = j + self.featureYsize/2

		if xw0 < 0:
			xw0 = 0
			xw1 = self.featureXsize

		if xw1 > mx:
			xw0 -= xw1 - mx
			xw1 = mx - 1

		if yw0 < 0:
			yw0 = 0
			yw1 = self.featureYsize

		if my < yw1:
			yw0 -= yw1 - my
			yw1 = my - 1

		featurePoints = [xw0, yw0, xw1, yw1]
		featurePoints = [int(value) for value in featurePoints]
		return featurePoints

class NegriMatcher:
	def match(descriptors1, descriptors2):
		
		
		for index, des1 in enumerate(descriptors1):
			for index, des2 in enumerate(descriptors2):


def loadImage(result):
	if result.get("image64") is not None:
		img = b64decode(result['image64'])
		npimg = np.fromstring(img, dtype=np.uint8) 
		cvImage = cv.imdecode(npimg, 1)
		print(result['date'])
		return cvImage
	else:
		return None

def getMatches(detector, matcher, img1, img2):
	(kp1, des1) = detector.detectAndCompute(img1, None)
	(kp2, des2) = detector.detectAndCompute(img2, None)
	matches = matcher.match(des1, des2)

	return extractVectors(matches, kp1, kp2)

def runSerie(detector, matcher, docs):
	serieResult = []

	totalDocs = docs.count(True)

	if totalDocs <= 0:
		return serieResult

	img1 = loadImage(docs.next())
	for index in range(0, totalDocs-1):
		img2 = loadImage(docs.next())
		vectors = getMatches(detector, matcher, img1, img2)
		serieResult.append(vectors)
		img1 = img2

	return serieResult

def extractVectors(matches, kp1, kp2):
	X, Y, U, V = [], [], [], []

	for match in matches:
		i1, i2 = match.queryIdx, match.trainIdx
		
		p1 = kp1[i1].pt
		p2 = kp2[i2].pt
		
		X.append(p1[0])
		Y.append(p1[1])

		U.append(p2[0])
		V.append(p2[1])

	X = np.array(X)
	Y = np.array(Y)
	U = np.array(U)
	V = np.array(V)
	return np.array([X, Y, U, V])

def getImagesFromDB(collection, limit):
	return collection.find({"type": "vis"}).sort("date", DESCENDING).limit(limit)

def getImageDimessions(doc):
	im = loadImage(doc.next())
	return im.shape

def plotVectorMap(data, title, index):
	X, Y, U, V = data

	plt.title(title + " - " + str(index))
	plt.quiver(X, Y, U, V, color='r')
	plt.show()

def plotSet(fieldSet, title):
	for index, field in enumerate(fieldSet):
		plotVectorMap(field, title, index)

def getNegriDetector():
	# Ideia to reproduce Negri method to recognize similar matrices
	return NegriDetector(85, 255, 6, 6)

def getNegriMatcher():
	pass

def runExperiment(collection, serieSize):
	# matcher = cv.BFMatcher(crossCheck=True)
	# siftDetector = cv.xfeatures2d.SIFT_create()
	# surfDetector = cv.xfeatures2d.SURF_create()
	# orbDetector = cv.ORB_create()

	negriDetector = getNegriDetector()
	negriMatcher = getNegriMatcher()

	imageDocs = getImagesFromDB(collection, serieSize)

	negriFields = runSerie(negriDetector, negriMatcher, imageDocs)
	# siftFields = runSerie(siftDetector, matcher, imageDocs)
	# surfFields = runSerie(surfDetector, matcher, imageDocs)

	# orbMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# orbFields = runSerie(orbDetector, orbMatcher, imageDocs)
	# plotSet(surfFields, "SURF")

	return None

def main():
	client = MongoClient('192.168.0.16', 27017)
	imagesCollection = client["nephos-comparation"]["images"]
	results = runExperiment(imagesCollection, 2)

	client.close()
	return results

# Query images
# Find features and make matches
# Filter matches
# plot map

surfFields = main()