import cv2 as cv
import sys
from util import basicFormater
from util import loadFiles
from util import seekMatches
from util import printer
from util import countKeypointsByPhoto
from util import processingTimeByPhoto
from util import countMatchesByPair
from util import stdByPhoto
from util import meanByPhoto

path = sys.argv[1]
# path = '../resource/buffer/test'
filesList = loadFiles(path)

matcher = cv.BFMatcher(crossCheck=True)
detector = cv.xfeatures2d.SURF_create()

# print stdByPhoto(filesList, path)
print meanByPhoto(filesList, path)
# matches_set = seekMatches(detector, matcher, filesList, path)
# filtered = basicFormater(matches_set)
# print printer(filtered)
# print countKeypointsByPhoto(matches_set)
# print processingTimeByPhoto(matches_set)
# print countMatchesByPair(matches_set)
# print matches_set[0]['matches']