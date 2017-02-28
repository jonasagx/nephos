import sys
import csv
import cv2 as cv
import numpy as np

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

def getCollection(host, port, dbName, collectionName):
	client = MongoClient(host, port)
	return client[dbName][collectionName]

def loadImage(cursor):
	result = cursor.next()

	if result.get("image64") is not None:
		img = b64decode(result['image64'])
		npimg = np.fromstring(img, dtype=np.uint8) 
		cvImage = cv.imdecode(npimg, 1)
		print(result['date'])
		return cvImage
	else:
		return None


def seekMatches(detector, matcher, img1, img2):
	(kp1, des1) = detector.detectAndCompute(img1, None)
	(kp2, des2) = detector.detectAndCompute(img2, None)

	return matcher.match(des1, des2)


def runSerie(detector, matcher, docs, algorithmName):
	serieResult = []

	totalDocs = docs.count()
	img1 = loadImage(docs)
	for index in range(0, totalDocs-1):
		img2 = loadImage(docs)
		matches = seekMatches(detector, matcher, img1, img2)
		serieResult.append(matches)
		img1 = img2

	return serieResult

def runExperiment(imagesDb)
	matcher = cv.BFMatcher(crossCheck=True)
	siftDetector = cv.xfeatures2d.SIFT_create()
	surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	# runSerie(siftDetector, matcher, imagesDb, "sift")
	# runSerie(surfDetector, matcher, imagesDb, "surf")
	runSerie(orbDetector, matcher, imagesDb, "orb")

def main():
	imagesCollection = getCollection('192.168.0.16', 27017, "nephos-test", "images")

	imagesDb = imagesCollection.find({"type": "rb"}).sort({"date":DESCENDING})
	runExperiment(imagesDb)

# Query images
# Find features and make matches

# Filter matches

# plot map
main()