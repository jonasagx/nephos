import sys
import csv

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from base64 import b64decode
from pymongo import DESCENDING
from pymongo import MongoClient

from util import printer
# from util import loadFiles
from util import saveToFile
# from util import validSerie
# from util import seekMatches
from util import matchPrinter
from util import basicFormater
from util import globalPrinter
from util import countPositives
from util import countMatchesByPair
from util import countKeypointsByPhoto
from util import processingTimeByPhoto

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

		# size1 = kp1[i1].size
		# size2 = kp2[i2].size
		# angle1 = kp1[i1].angle
		# angle2 = kp2[i2].angle
		# vectors.append((p1, p2, (size1, size2), (angle1, angle2)))
		# vectors.append((p1, p2))
	return [X, Y, U, V]

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

def runExperiment(collection):
	matcher = cv.BFMatcher(crossCheck=True)
	siftDetector = cv.xfeatures2d.SIFT_create()
	surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	imageDocs = getImagesFromDB(collection, 5)

	# siftFields = runSerie(siftDetector, matcher, imageDocs)
	surfFields = runSerie(surfDetector, matcher, imageDocs)
	# orbFields = runSerie(orbDetector, matcher, imageDocs)
	plotSet(surfFields, "SURF")

	return surfFields

def main():
	client = MongoClient('192.168.0.16', 27017)
	imagesCollection = client["nephos-test"]["images"]
	results = runExperiment(imagesCollection)

	client.close()
	return results

# Query images
# Find features and make matches
# Filter matches
# plot map

surfFields = main()