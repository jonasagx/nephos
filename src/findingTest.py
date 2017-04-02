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

from datetime import datetime
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
		serie.append(len(kp1))
	return serie

def convertToImage(doc):
	img = base64.b64decode(doc['image64'])
	npimg = np.fromstring(img, dtype=np.uint8)
	# print(datetime.fromtimestamp(doc['date']))
	return cv.imdecode(npimg, 1)

def convertToImages(imagesFromDb):
	cvImages = []

	firstDoc = imagesFromDb.next()
	cvImages.append(convertToImage(firstDoc))
	timestamp = firstDoc['date']

	for result in imagesFromDb:
		source = convertToImage(result)
		cvImages.append(source)
	return (timestamp, cvImages)

def runExperiment(type, collection, size):
	series = {}
		
	results = collection.find({"type":type}).sort("date").limit(size)
	firstDate, cvImages = convertToImages(results)
	algorithms = []

	algorithms.append(Algorithm("SIFT", cv.xfeatures2d.SIFT_create()))
	algorithms.append(Algorithm("SURF", cv.xfeatures2d.SURF_create()))
	algorithms.append(Algorithm("ORB", cv.ORB_create()))
	algorithms.append(Algorithm("NEGRI", NegriDetector(80, 255)))

	series.update(runSerieExtraction(cvImages, algorithms))

	df = createDataFrame(series, firstDate, "45min", size)
	plotDataFrame(df)

	return df

def createDataFrame(data, firstDate, frequency, size):
	ts = pd.DataFrame.from_dict(data)
	date = datetime.fromtimestamp(firstDate)

	return ts.set_index(pd.date_range(date, periods=size, freq=frequency))

def plotDataFrame(df):
	ax = df.plot(figsize=(15, 5), logy=True)
	ax.set_xlabel("Tempo")
	ax.set_ylabel("key-points/imagem - escala log")
	# plt.title("Pontos de interesse por imagem")
	plt.savefig("serie_keypoints.png", dpi = 600)
	plt.show()

def main():
	imagesCollection = getCollection('192.168.0.16', 27017, "nephos", "images")
	
	# types = imagesCollection.find({}).distinct("type")
	# types = imagesCollection.find({"type": "vis"}).sort("date").limit(3)
	return runExperiment('ir2', imagesCollection, 300)

ts = main()
