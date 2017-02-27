'''
	Script to test if the algorithms can find features on different images.
	load picures
	for each algorithm try to extract features from all the pictures
	show how many features were found from each picture
'''


import base64
import pprint

import cv2 as cv
import numpy as np

from pymongo import MongoClient

class Algorithm:
	def __init__(self, name, detector):
		self.name = name
		self.detector = detector

def getCollection(host, port, dbName, collectionName):
	client = MongoClient(host, port)
	return client[dbName][collectionName]

def runSerieExtraction(cvImages, algorithms):
	results = {}

	for algorithm in algorithms:
		results[algorithm.name] = runSerie(cvImages, algorithm.detector)

	return results

def runSerie(cvImages, detector):
	serie = []

	for index, image in enumerate(cvImages):
		(kp1, des1) = detector.detectAndCompute(image, None)
		serie.append(len(kp1))
	return serie

def convertToImages(imagesFromDb):
	cvImages = []

	for result in imagesFromDb:
		img = base64.b64decode(result['image64'])
		npimg = np.fromstring(img, dtype=np.uint8) 
		source = cv.imdecode(npimg, 1)
		cvImages.append(source)
	return cvImages

def runExperiment1(types, collection):
	experimentResults = {}
	for t in types:	
		results = collection.find({"type":t})
		cvImages = convertToImages(results)
		algorithms = []

		algorithms.append(Algorithm("SIFT" + "_" + t, cv.xfeatures2d.SIFT_create()))
		algorithms.append(Algorithm("SURF" + "_" + t, cv.xfeatures2d.SURF_create()))
		algorithms.append(Algorithm("ORB" + "_" + t, cv.ORB_create()))

		experimentResults.update(runSerieExtraction(cvImages, algorithms))
	return experimentResults

def main():
	imagesCollection = getCollection('192.168.0.16', 27017, "nephos-test", "images")
	
	types = imagesCollection.find({}).distinct("type")
	return runExperiment1(types, imagesCollection)

experimentResults = main()
