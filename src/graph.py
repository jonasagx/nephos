import sys
import cv2 as cv
from util import printer
from util import loadFiles
from util import validSerie
from util import seekMatches
from util import matchPrinter
from util import basicFormater
from util import globalPrinter
from util import countPositives
from util import countMatchesByPair
from util import countKeypointsByPhoto
from util import processingTimeByPhoto
from util import flatView

path = sys.argv[1]
rawFilesList = loadFiles(path)
filesList = validSerie(rawFilesList)

matcher = cv.BFMatcher(crossCheck=True)
# detector = cv.xfeatures2d.SURF_create()
detector = cv.xfeatures2d.SIFT_create()
# detector = cv.ORB_create()

matches_set = seekMatches(detector, matcher, filesList, path)
filtered = basicFormater(matches_set)
flat = flatView(filtered)
print printer(flat)
# print globalPrinter(matches_set, filesList, path)
# print matchPrinter(filtered)


