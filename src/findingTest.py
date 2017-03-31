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
import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient

class Algorithm:
	def __init__(self, name, detector):
		self.name = name
		self.detector = detector

class NegriDetector:
	def __init__(self, minValue, maxValue):
		self.minValue = minValue
		self.maxValue = maxValue

	def detectAndCompute(self, im, something=None):
		imFlat = im.flatten()
		size = ((imFlat >= self.minValue) & (imFlat <= self.maxValue)).sum()
		return (np.zeros(size), None)

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
		l = len(kp1)
		serie.append(np.log10(l))
	return serie

def convertToImages(imagesFromDb):
	cvImages = []

	for result in imagesFromDb:
		img = base64.b64decode(result['image64'])
		npimg = np.fromstring(img, dtype=np.uint8) 
		source = cv.imdecode(npimg, 1)
		cvImages.append(source)
	return cvImages

def runExperiment(types, collection):
	experimentResults = {}
	for t in types:	
		results = collection.find({"type":t}).sort("date").limit(100)
		cvImages = convertToImages(results)
		algorithms = []

		algorithms.append(Algorithm("SIFT", cv.xfeatures2d.SIFT_create()))
		algorithms.append(Algorithm("SURF", cv.xfeatures2d.SURF_create()))
		algorithms.append(Algorithm("ORB", cv.ORB_create()))
		algorithms.append(Algorithm("NEGRI", NegriDetector(80, 255)))

		experimentResults.update(runSerieExtraction(cvImages, algorithms))

	return plotDataFrame(experimentResults)

def plotDataFrame(data):
	ts = pd.DataFrame.from_dict(data)
	size = ts[ts.columns[0]].size
	ts.set_index(pd.date_range('1/1/2011 00:00:00', periods=size, freq='45min')).plot()
	plt.show()

	return ts

def main():
	imagesCollection = getCollection('192.168.0.16', 27017, "nephos", "images")
	
	# types = imagesCollection.find({}).distinct("type")
	# types = imagesCollection.find({"type": "vis"}).sort("date").limit(3)
	types = ['ir2']
	return runExperiment(types, imagesCollection)

ts = main()
