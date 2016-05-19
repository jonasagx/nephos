# -*- coding: utf-8 -*-
import csv
import sys

import numpy as np

from scipy.spatial import distance

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth

vectorFile = sys.argv[1]
dots = []

# zero = (0., 0, 0, 0, 0)

with open(vectorFile, 'rb') as csvfile:
	vectorData = csv.reader(csvfile, delimiter=',')
	next(vectorData, None) #Skips the header
	for kpt in vectorData:
		dots.append( [int(kpt[0]), int(kpt[1]), int(kpt[2]), int(kpt[3]), float(kpt[5]), int(kpt[6])] )

X = np.array(dots)

print(X)

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

###############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()