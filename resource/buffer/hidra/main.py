# input = list()
# nodes = ()
# keys = list()

# loader = None
# spreader = None
# matcher = None

from multiprocessing import Pool
import cv2 as cv
from lib import *

path = "/home/jonas/apps/nephos/test/"

inputs = loadPictures(path)
pics = readPictures(inputs, path)
pool = Pool(processes = 4)

result = pool.map(detect, pics)