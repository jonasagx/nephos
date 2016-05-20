import os
import cv2 as cv
import sys
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
    #print 'seekMatches'
    trackSerie = {}
    detector = cv.BRISK_create()
    '''
    If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
'''
    matcher = cv.BFMatcher()

    for index in xrange(len(filesList) - 1):
        #print index, index + 1, filesList[index], filesList[index + 1]
        #print path + filesList[index], path + filesList[index + 1]
        img1 = cv.imread(path + filesList[index], 0)
        img2 = cv.imread(path + filesList[index + 1], 0)

        (kp1, des1) = detector.detectAndCompute(img1, None)
        (kp2, des2) = detector.detectAndCompute(img2, None)

        matches = matcher.match(des1, des2)

        trackSerie[index] = {'matches': matches,
                              'kp1': kp1,
                              'kp2': kp2,
                              'des1': des1,
                              'des2': des2}
    return trackSerie

def printer(lines):
    print("x1, y1, x2, y2")
    for line in lines:
        s = "%d, %d, %d, %d" % (line[0][0], line[0][1], line[1][0], line[1][1])
        print(s)

path = ''
filesList = []
eucDistances = []
distances = []
matches_set = {}
filtered = []

if __name__ == '__main__':
    path, filesList = loadFiles()
    #print len(filesList), path
    matches_set = seekMatches()
    filtered = filterEuDistances(matches_set)
    printer(filtered)
