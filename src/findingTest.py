'''
	Script to test if the algorithms can find features on different images.
	load picures
	for each algorithm try to extract features from all the pictures
	show how many features were found from each picture
'''


import time
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

	def detect(self, im, something=None):
		points = []
		imGray = im[:, :, 0]

		for i in range(len(imGray)):
			for j in range(len(imGray[0])):
				if imGray[i, j] >= self.minValue and imGray[i, j] <= self.maxValue:
					points.append((i,j))

		return points


	def detectAndCompute(self, im, something=None):
		points = []
		imGray = im[:, :, 0]

		for i in range(len(imGray)):
			for j in range(len(imGray[0])):
				if imGray[i, j] >= self.minValue and imGray[i, j] <= self.maxValue:
					points.append((i,j))

		return points, None

def getCollection(host, port, dbName, collectionName):
	client = MongoClient(host, port)
	return client[dbName][collectionName]

def runSerieExtraction(cvImages, algorithms):
	results = {}

	for algorithm in algorithms:
		keyPoints, times = runSerie(cvImages, algorithm.detector)
		results[algorithm.name] = {'keypoints': keyPoints, 'times': times}

	return results

def timer(t0=None):
	if t0 == None:
		return time.time()
	else:
		return time.time() - t0

def runSerie(cvImages, detector):
	serie = []
	times = []

	for index, image in enumerate(cvImages):
		t0 = timer()
		# (kp1, des1) = detector.detectAndCompute(image, None)
		kp1 = detector.detect(image)
		serie.append(len(kp1))
		times.append(timer(t0))
	return (serie, times)

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

	keyPoints = createDataFrameKeyPoints(series, firstDate, "45min", size)
	plotKeyPoints(keyPoints)

	times = createDataFrameTimes(series, firstDate, "45min", size)
	plotTimes(times)

def createDataFrameTimes(data, firstDate, frequency, size):
	series = {}
	for k, v in data.items():
		series[k] = v['times']

	ts = pd.DataFrame.from_dict(series)
	date = datetime.fromtimestamp(firstDate)

	return ts.set_index(pd.date_range(date, periods=size, freq=frequency))

def createDataFrameKeyPoints(data, firstDate, frequency, size):
	series = {}
	for k, v in data.items():
		series[k] = v['keypoints']

	ts = pd.DataFrame.from_dict(series)
	date = datetime.fromtimestamp(firstDate)

	return ts.set_index(pd.date_range(date, periods=size, freq=frequency))

def plotTimes(times):
	ax = times.plot(figsize=(15, 5), logy=True)
	ax.set_xlabel("Data")
	ax.set_ylabel("Tempo de processamento (s)")
	# plt.title("Pontos de interesse por imagem")
	plt.savefig("serie_times.png", dpi = 600)
	plt.show()

def plotKeyPoints(df):
	ax = df.plot(figsize=(15, 5), logy=True)
	ax.set_xlabel("Data")
	ax.set_ylabel("Pontos de interesse - escala log")
	# plt.title("Pontos de interesse por imagem")
	plt.savefig("serie_keypoints.png", dpi = 600)
	plt.show()

def main():
	imagesCollection = getCollection('192.168.0.16', 27017, "nephos", "images")
	
	# types = imagesCollection.find({}).distinct("type")
	# types = imagesCollection.find({"type": "vis"}).sort("date").limit(3)
	return runExperiment('ir2', imagesCollection, 400)

main()
