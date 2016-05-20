import cv2 as cv
import sys
from util import basicFormater
from util import loadFiles
from util import seekMatches
from util import printer
from util import countKeypointsByPhoto
from util import processingTimeByPhoto
from util import countMatchesByPair
from util import globalPrinter
from util import matchPrinter
from util import countPositives

path = sys.argv[1]
# path = '../resource/buffer/test'
filesList = loadFiles(path)

matcher = cv.BFMatcher(crossCheck=True)
detector = cv.xfeatures2d.SURF_create()
# detector = cv.ORB_create()

matches_set = seekMatches(detector, matcher, filesList, path)
filtered = basicFormater(matches_set)
# print filtered
print printer(filtered)
# print globalPrinter(matches_set, filesList, path)
# print matchPrinter(filtered)


