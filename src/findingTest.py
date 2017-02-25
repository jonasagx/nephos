'''
	Script to test if the algorithms can find features on different images
'''

import cv2 as cv
import base64
import numpy as np
from pymongo import MongoClient

def getCollection(host, port, dbName, collectionName):
	client = MongoClient(host, port)
	return client[dbName][collectionName]

def runExperiment(imagesFromDb, detector):
	for result in imagesFromDb:
		img = base64.b64decode(result['image64'])
		npimg = np.fromstring(img, dtype=np.uint8) 
		source = cv.imdecode(npimg, 1)
		(kp1, des1) = detector.detectAndCompute(source, None)

	print(len(kp1))
	# for each algorithm try to extract features from all the pictures
	# show how many features were found from each picture

def main():
	# load picures
	images = getCollection('192.168.0.16', 27017, "nephos-test", "images")
	results = images.find({"type":"rb"})

	siftDetector = cv.xfeatures2d.SIFT_create()
	runExperiment(results, siftDetector)

main()