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

# from util import printer
# from util import loadFiles
from util import saveToFile
# from util import validSerie
# from util import seekMatches
# from util import matchPrinter
# from util import basicFormater
# from util import globalPrinter
# from util import countPositives
# from util import countMatchesByPair
# from util import countKeypointsByPhoto
# from util import processingTimeByPhoto

class Detector:
	def __init__(self):
		pass

	def detectAndCompute(self, im, noneValue):
		pass

class NegriDetector(Detector):
	def detectAndCompute(self, im, noneValue):
		labelObjects, number_of_objects = ndimage.label(im)
		areasOfInterest = ndimage.find_objects(labelObjects)

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

def runExperiment(collection, serieSize):
	matcher = cv.BFMatcher(crossCheck=True)
	siftDetector = cv.xfeatures2d.SIFT_create()
	surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	negriDetector = getNegriDetector()

	imageDocs = getImagesFromDB(collection, serieSize)

	# siftFields = runSerie(siftDetector, matcher, imageDocs)
	surfFields = runSerie(surfDetector, matcher, imageDocs)
	# orbFields = runSerie(orbDetector, matcher, imageDocs)
	plotSet(surfFields, "SURF")

	return surfFields

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

def getDistancesFromSerie(serie):
	ds = []

	for x1, y1, x2, y2 in serie:
		d = distance.euclidian((x1, y1),(x2, y2))
		ds.append(d)
	return distances
