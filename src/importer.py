'''
	This script imports a batch of images to a mongoDB collection.

	Image structure:
		* image as base64 string
		* date
		* type
'''

import os
import sys
import base64
import datetime

from PIL import Image
from pymongo import MongoClient

def loadFiles(path, fileExtention):
	assert len(path) > 0, "Pass folder path as param"
	if not path.endswith("/"):
		path += "/"
	filesList = os.listdir(path)
	
	for file in filesList:
		if not file.endswith(fileExtention):
			filesList.remove(file)
	
	filesList.sort()
	return filesList

def getCollection(host, port, dbName, collectionName):
	client = MongoClient(host, port)
	return (client, client[dbName][collectionName])

def getDate(stringDate):
	year = int(stringDate[:4])

	#Minus 1 because baseDate has to start on day 1. Otherwise it would pass one day
	dayInYear = int(stringDate[4:7]) - 1
	hour = int(stringDate[8:10])
	minutes = int(stringDate[10:12])

	baseDate = datetime.datetime(year, 1, 1, hour, minutes, 0)
	timeDelta = datetime.timedelta(days=dayInYear)

	dt = baseDate + timeDelta
	return dt.timestamp()

def getImageType(filename):
	return filename.split('.')[0][12:]

def convertImageToBase64(filePath):
	return base64.b64encode(open(filePath, "rb").read())

def getImageDoc(filePath, tempFilePath):
	filename = filePath.split('/')[-1]

	doc = {}
	doc['date'] = getDate(filename)
	doc['type'] = getImageType(filename)
	doc['image64'] = convertImageToBase64(tempFilePath)

	return doc

def main():
	path = sys.argv[1]
	filesList = loadFiles(path, ".jpg")
	temFile = "/tmp/nephos_temp.jpg"
	cropSquare = [79, 109, 509, 346]
	imageBulk = []

	for filename in filesList:
		# load image
		print(filename)
		im = Image.open(path + filename)
		
		# crop it
		im.crop(cropSquare).save(temFile)
		im.close()

		# extract info from name
		imDoc = getImageDoc(filename, temFile)
		imageBulk.append(imDoc)

	cli, images = getCollection('localhost', 27017, "nephos-comparation", "images")
	images.insert_many(imageBulk)
	cli.close()

main()