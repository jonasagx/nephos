import sys
import csv
import cv2 as cv
from util import printer
from util import loadFiles
from util import saveToFile
from util import validSerie
from util import seekMatches
from util import matchPrinter
from util import basicFormater
from util import globalPrinter
from util import countPositives
from util import countMatchesByPair
from util import countKeypointsByPhoto
from util import processingTimeByPhoto

def main():
	path = sys.argv[1]
	rawFilesList = loadFiles(path)
	filesList = validSerie(rawFilesList)

	matcher = cv.BFMatcher(crossCheck=True)

	siftDetector = cv.xfeatures2d.SIFT_create()
	surfDetector = cv.xfeatures2d.SURF_create()
	orbDetector = cv.ORB_create()

	runExperiemnt(siftDetector, matcher, filesList, path, "sift")
	runExperiemnt(surfDetector, matcher, filesList, path, "surf")
	runExperiemnt(orbDetector, matcher, filesList, path, "orb")


def runExperiemnt(detector, matcher, filesList, path, algorithmName):
	matches_set = seekMatches(detector, matcher, filesList, path)
	filtered = basicFormater(matches_set)

	dataFiltered = printer(filtered)
	saveToFile(algorithmName + "-filtered.csv", dataFiltered)

	dataGlobal = globalPrinter(matches_set, filesList, path)
	saveToFile(algorithmName + "-global.csv", dataGlobal)

	dataMatch = matchPrinter(filtered)
	saveToFile(algorithmName + "-match.csv", dataMatch)

main()