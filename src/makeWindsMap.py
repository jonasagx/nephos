import sys
import csv
import time
import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from base64 import b64decode
from pymongo import DESCENDING
from pymongo import MongoClient
from scipy.spatial import distance
from scipy.stats import pearsonr

class Serie:
	def __init__(self, field, dates, types, algorithmName):
		self.field = field
		self.dates = dates
		self.types = types
		self.algorithmName = algorithmName

	def getPrettyTitle(self):
		title = self.timestamp2StringDate(self.dates[0])
		title += "-" + self.timestamp2StringDate(self.dates[1])
		title += " " + self.types[0]
		title += "-" + self.types[0]
		title += " " + self.algorithmName
		return title

	def getField(self):
		return self.field

	def timestamp2StringDate(self, timestamp):
		date = datetime.datetime.fromtimestamp(timestamp)
		return date.strftime('%d-%m-%Y %H:%M')

class Match:
	def __init__(self, queryIdx, trainIdx):
		self.queryIdx = queryIdx
		self.trainIdx = trainIdx

	def __str__(self):
		return str(self.queryIdx) + " " + str(self.trainIdx)

	def __repr__(self):
		return self.__str__()

class KeyPoint:
	def __init__(self, x, y, size):
		self.pt = [x, y]
		self.size = size

	def __str__(self):
		return str(self.pt[0]) + " " + str(self.pt[1])

	def __repr__(self):
		return self.__str__()

class NegriDetector:
	percentage = 0.3
	def __init__(self, min, max, featureXsize, featureYsize):
		# grayscale threshold
		self.min = min
		self.max = max

		self.checkStepSize(featureXsize)
		self.checkStepSize(featureYsize)

		self.xStep = int(featureXsize * self.percentage)
		self.yStep = int(featureYsize * self.percentage)

		# feature window dimensions
		if not (featureXsize % 2 == 0 and featureYsize % 2 == 0):
			raise Exception("Dimensions should both be even")

		self.featureXsize = featureYsize
		self.featureYsize = featureYsize

	def checkStepSize(self, featureSize):
		step = int(featureSize * self.percentage)
		if step <= 0:
			raise Exception("Step must be an Integer bigger than zero. Received: ", step)

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

		for i in range(0, X, self.xStep):
			for j in range(0, Y, self.yStep):
				if grey.item(i, j) == self.max:
					feature = self.takeFeature(grey, i, j)
					descriptors.append(feature)
					keypoints.append(KeyPoint(j, i, self.min))

		return (keypoints, descriptors)

	def takeFeature(self, im, i, j):
		xw0, yw0, xw1, yw1 = self.getFeatureWindow(i, j, im)
		feature = im[xw0:xw1, yw0:yw1]

		if feature.shape != (self.featureXsize, self.featureYsize):
			raise Exception((xw0, yw0, xw1, yw1), feature.shape)
		return feature

	def getFeatureWindow(self, i, j, im):
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
		# self.plotMatrix(corrMatrix)
		return self.findBestMatches(corrMatrix)

	def findBestMatches(self, corrMatrix):
		matches = []
		for index, row in enumerate(corrMatrix):
			maxRowIndex, maxFromRow = self.getMaxAndIndex(row)

			column = [c[maxRowIndex] for c in corrMatrix]
			maxColumnIndex, maxFromColumn = self.getMaxAndIndex(column)

			if maxFromRow == maxFromColumn:
				m = Match(maxColumnIndex, maxRowIndex)
				matches.append(m)

		return matches

	def getMaxAndIndex(self, array):
		maxValue = max(array)
		maxIndex = array.index(maxValue)

		return (maxIndex, maxValue)

	def getCorrCoefs(self, descriptors1, descriptors2):
		corrCoef, row = [], []
		for des1 in descriptors1:
			for des2 in descriptors2:
				coef, pValue = pearsonr(des1.flatten(), des2.flatten())
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
		return cvImage
	else:
		return None

def getMatches(detector, matcher, img1, img2):
	(kp1, des1) = detector.detectAndCompute(img1, None)
	(kp2, des2) = detector.detectAndCompute(img2, None)
	matches = matcher.match(des1, des2)

	return extractVectors(matches, kp1, kp2)

def timer(t0=None):
	if t0 == None:
		return time.time()
	else:
		return time.time() - t0

def runSerie(detector, matcher, docs, algorithmName):
	t0 = timer()
	serieResult = []
	docs.rewind()
	totalDocs = docs.count(True)

	if totalDocs <= 0:
		return serieResult

	docIm1 = docs.next()
	img1 = loadImage(docIm1)
	for index in range(0, totalDocs-1):
		docIm2 = docs.next()
		img2 = loadImage(docIm2)

		vectors = getMatches(detector, matcher, img1, img2)
		serie = Serie(vectors, (docIm1['date'], docIm2['date']), (docIm1['type'], docIm2['type']), algorithmName)
		serieResult.append(serie)

		img1 = img2
		docIm1 = docIm2

	print("%.3fs - %s" % (timer(t0), algorithmName))
	return serieResult

def extractVectors(matches, kp1, kp2):
	X, Y, U, V = [], [], [], []

	for match in matches:
		queryId, trainId = match.queryIdx, match.trainIdx
		
		p1 = kp1[queryId].pt
		p2 = kp2[trainId].pt
		
		X.append(p1[0])
		Y.append(p1[1])

		U.append(p2[0])
		V.append(p2[1])

	return np.array([X, Y, U, V])

def getImagesFromDB(collection, limit):
	return collection.find({"type": "vis"}).sort("date").skip(11).limit(limit)

def getImageDimessions(doc):
	im = loadImage(doc.next())
	return im.shape

def getHumanTitle(timestamp, imgType):
	date = datetime.datetime.fromtimestamp(timestamp)
	humanDate = date.strftime('%d-%m-%Y %H:%M')

	return humanDate + " - " + imgType

def plotMatrix(matrix, title):
	plt.title(title)
	plt.imshow(matrix)
	plt.savefig(str(title) + ".png", dpi = 600)
	plt.show()

def plotVectorMap(data, title):
	X, Y, U, V = data

	plt.title(title)
	plt.quiver(X, Y, U, V, units='xy')
	plt.savefig(title + ".png", dpi = 600)
	plt.show()

def plotWindFields(fieldSet):
	for serie in fieldSet:
		plotVectorMap(serie.field, serie.getPrettyTitle())

def plotImageSet(docs):
	docs.rewind()

	for index  in range(docs.count(True)):
		doc = docs.next()
		image = loadImage(doc)
		title = getHumanTitle(doc['date'], doc['type'])
		plotMatrix(image, title)

def getNegriDetector():
	# Ideia to reproduce Negri method to recognize similar matrices
	return NegriDetector(70, 255, 50, 50)

def getNegriMatcher():
	return NegriMatcher()

def runExperiment(collection, serieSize):
	matcher = cv.BFMatcher(crossCheck=True)
	siftDetector = cv.xfeatures2d.SIFT_create()
	surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	negriDetector = getNegriDetector()
	negriMatcher = getNegriMatcher()

	imageDocs = getImagesFromDB(collection, serieSize)

	negriFields = runSerie(negriDetector, negriMatcher, imageDocs, "NEGRI")
	siftFields = runSerie(siftDetector, matcher, imageDocs, "SIFT")
	surfFields = runSerie(surfDetector, matcher, imageDocs, "SURF")

	orbMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	orbFields = runSerie(orbDetector, orbMatcher, imageDocs, "ORB")
	
	# Ploting
	plotImageSet(imageDocs)

	plotWindFields(negriFields)
	plotWindFields(orbFields)
	plotWindFields(surfFields)
	plotWindFields(siftFields)

	allResults = {
		"negri": negriFields, 
		"orb": orbFields,
		"surf": surfFields, 
		"sift": siftFields
	}

	showHomogeneity(allResults)

	return allResults

def plotHist(data, algorithm):
	# plt.vlines(t, [0], data)
	plt.hist(data, bins=100)
	plt.title("Tamanho dos vetores - " + algorithm)
	plt.xlabel("Tamanho")
	plt.ylabel("Frequência")
	plt.show()

def showHomogeneity(allResults):
	for resultKey in allResults:
		series = allResults[resultKey]
		length = []
		for serie in series:

			x, y, u, v = serie.field
			vectorsZip = zip(x,y,u,v)
			vectors = list(vectorsZip)
			for v in vectors:
				a,b,c,d = v
				d = distance.euclidean((a,b), (c,d))
				# an = np.arctan(shiftVector(a,b,c,d))
				length.append(d)
		plotHist(length, resultKey)

def main():
	# client = MongoClient('192.168.0.16', 27017)
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