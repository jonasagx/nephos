# -*- coding: utf-8 -*-
import csv
import sys

import numpy as np
from scipy import stats
from scipy.spatial import distance

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

vectorFile = sys.argv[1]
dots = []

zero = (0., 0, 0, 0, 0)

with open(vectorFile, 'rb') as csvfile:
	vectorData = csv.reader(csvfile, delimiter=',')
	next(vectorData, None) #Skips the header
	for kpt in vectorData:
		dots.append( (float(kpt[0]), float(kpt[1]), float(kpt[2]), float(kpt[3]), float(kpt[4]), float(kpt[5]), float(kpt[6])) )

dis = []

origin = (0,0,0,0,0,0,0)
for dot in dots:
	# print dot
	# print "%.3f," % distance.euclidean(origin, dot)
	dis.append(distance.euclidean(origin, dot))

import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)

# # x = np.random.normal(0,1,1000)
# numBins = 20
# ax.hist(dis, numBins, color='green', alpha=0.8)
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
res = stats.probplot(dis, dist=stats.loggamma, sparams=(2.5,), plot=ax)
ax.set_title("Probplot")
plt.show()