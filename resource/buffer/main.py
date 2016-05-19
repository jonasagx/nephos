import sys
import os
import cv2 as cv
import numpy as np

def loadPicture(path):
    listFiles = os.listdir(path)
    listFiles.sort()

    listPics = []

    for p in listFiles:
        #print path + p
        #print cv.imread(path + p, 0)
        listPics.append( filter(cv.imread(path + p, 0), 100, 70) )

    return listPics

def filter(im, max, min):
    x, y = im.shape
    for i in range(x):
        for j in range(y):
            if im.item(i, j) > min and im.item(i, j) < max:
                pass
            else:
                im.itemset(i, j, 0)
    return im

def sumUp(listPics):
    s = np.zeros(listPics[0].shape)
    for index3, value in enumerate(listPics):
        s = s + value
        print 100 * np.count_nonzero(value)/ value.size, "%", index
    cv.imwrite("sumM.png", s)
        
pics = []

if __name__ == '__main__':
    pics = loadPicture(sys.argv[1])
    sumUp(pics)
