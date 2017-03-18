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
from scipy.stats import pearsonr

class Match:
	def __init__(self, queryIdx, trainIdx):
		self.queryIdx = queryIdx
		self.trainIdx = trainIdx

class KeyPoint:
	def __init__(self, x, y, size):
		self.pt = [x, y]
		self.size = size

	def __str__(self):
		return str(self.pt[0]) + " " + str(self.pt[1])

	def __repr__(self):
		return self.__str__()

class NegriDetector:
	def __init__(self, min, max, featureXsize, featureYsize):
		# grayscale threshold
		self.min = min
		self.max = max
		self.xStep = int(featureXsize * 0.3)
		self.yStep = int(featureYsize * 0.3)

		# feature window dimensions
		if not (featureXsize % 2 == 0 and featureYsize % 2 == 0):
			raise Exception("Dimensions should both be even")

		self.featureXsize = featureYsize
		self.featureYsize = featureYsize

	def prepareImage(self, im):
		grey = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
		ret, th1 = cv.threshold(grey, self.min, self.max, cv.THRESH_BINARY)
		# th1 = im[:,:,0]
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

		for i in range(0, X, self.xStep):
			for j in range(0, Y, self.yStep):
				if grey.item(i, j) == self.max:
					feature = self.takeFeature(grey, i, j)
					descriptors.append(feature)
					keypoints.append(KeyPoint(i, j, self.min))

		return (keypoints, descriptors)

	def takeFeature(self, im, i, j):
		xw0, yw0, xw1, yw1 = self.checkFeatureWindow(i, j, im)

		feature = im[xw0:xw1, yw0:yw1]

		if feature.shape != (self.featureXsize, self.featureYsize):
			raise Exception((xw0, yw0, xw1, yw1), feature.shape)

		# im[xw0:xw1, yw0:yw1] = 0
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
			xw0 = xw0 - (xw1 - mx)
			xw1 = mx

		if yw0 < 0:
			yw0 = 0
			yw1 = self.featureYsize

		if yw1 > my:
			yw0 = yw0 - (yw1 - my)
			yw1 = my

		featurePoints = [xw0, yw0, xw1, yw1]
		featurePoints = [int(value) for value in featurePoints]
		return featurePoints

class NegriMatcher:
	def match(self, descriptors1, descriptors2):
		#Correlation coefs in matrix structure
		corrMatrix = self.getCorrCoefs(descriptors1, descriptors2)

		#Cross filter results
		self.plotMatrix(corrMatrix)
		return self.findBestMatches(corrMatrix)

	def findBestMatches(self, corrMatrix):
		matches = []
		for index, row in enumerate(corrMatrix):
			j, value = max( [(i, v) for i, v in enumerate(row)] )

			if row.count(value) == 1:
				matches.append( Match(index, j))

		return matches

	def plotMatrix(self, matrix):
		plt.imshow(matrix)
		plt.show()

	def getCorrCoefs(self, descriptors1, descriptors2):
		corrCoef, row = [], []
		for des1 in descriptors1:
			for des2 in descriptors2:
				coef, pValue = pearsonr(des1.flatten(), des2.flatten())
				# coef = np.sum((des1 - des2)**2)
				row.append(coef)
			corrCoef.append(row)
			# print(row)
			row = []
		return corrCoef

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
	docs.rewind()
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
	return NegriDetector(80, 255, 24, 24)

def getNegriMatcher():
	return NegriMatcher()

def runExperiment(collection, serieSize):
	matcher = cv.BFMatcher(crossCheck=True)
	# siftDetector = cv.xfeatures2d.SIFT_create()
	# surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	negriDetector = getNegriDetector()
	negriMatcher = getNegriMatcher()

	imageDocs = getImagesFromDB(collection, serieSize)

	negriFields = runSerie(negriDetector, negriMatcher, imageDocs)
	# siftFields = runSerie(siftDetector, matcher, imageDocs)
	# surfFields = runSerie(surfDetector, matcher, imageDocs)

	orbMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	orbFields = runSerie(orbDetector, orbMatcher, imageDocs)
	plotSet(negriFields, "NEGRI")
	plotSet(orbFields, "ORB")

	return {"negri": negriFields, "orb": orbFields}

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

results = main()