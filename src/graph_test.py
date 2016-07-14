import sys
import cv2 as cv
import numpy as np

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

from sklearn import metrics
from sklearn.cluster import KMeans

path = sys.argv[1]
rawFilesList = loadFiles(path)
filesList = validSerie(rawFilesList)

matcher = cv.BFMatcher(crossCheck=True)
# detector = cv.xfeatures2d.SIFT_create()
detector = cv.ORB_create()

matches_set = seekMatches(detector, matcher, filesList, path)
filtered = basicFormater(matches_set)
flat = flatView(filtered)
flat = np.asarray(flat)
# print printer(flat)

km = KMeans(n_clusters=13).fit(flat)
# print metrics.silhouette_score(flat, km.labels_, metric='euclidean')
results = metrics.silhouette_samples(flat, km.labels_)

clusters = {}
for index, label in enumerate(km.labels_):
	try:
		clusters[label].append({'index': index, 'silhouette': results[index]})
	except Exception, e:
		clusters[label] = [{'index': index, 'silhouette': results[index]}]
		
for cluster in clusters.keys():
	# print "Silhouette " + cluster + " " + np.mean([c['silhouette'] for c in clusters[cluster]])
	print np.mean([c['silhouette'] for c in clusters[cluster]])

# That's important, keep it
# print globalPrinter(matches_set, filesList, path)
# print matchPrinter(filtered)


