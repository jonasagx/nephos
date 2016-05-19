import os
import cv2 as cv
import sys
import csv
from filter import filterDistances
from filter import filterEuDistances

def loadFiles():
    assert len(sys.argv) > 1, "Pass folder path as param"
    path = sys.argv[1]
    assert path.find('/') != -1, "Pass a valid path with slash(/)"
    filesList = os.listdir(path)
    filesList.sort()
    return path, filesList

def seekMatches():
    trackSerie = {}
    #detector = cv.xfeatures2d.SURF_create()
    detector = cv.ORB_create()
    # detector = cv.xfeatures2d.SIFT_create()
    # detector = cv.xfeatures2d.StarDetector_create()
    '''
    If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
    '''
    matcher = cv.BFMatcher(crossCheck=True)

    for index in xrange(len(filesList)):
        if(filesList[index].find(".jpg") == -1):
            continue

        print path + filesList[index]
        img1 = cv.imread(path + filesList[index], 0)
        (kp1, des1) = detector.detectAndCompute(img1, None)
    return trackSerie

path = ''
filesList = []
eucDistances = []
distances = []
matches_set = {}
filtered = []

if __name__ == '__main__':
    path, filesList = loadFiles()
    print len(filesList), path
    matches_set = seekMatches()