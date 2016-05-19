# -*- coding: utf-8 -*-
import csv
import sys

import numpy as np

from scipy.spatial import distance

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

vectorFile = sys.argv[1]
dots = []

zero = (0., 0, 0, 0, 0)

with open(vectorFile, 'rb') as csvfile:
	vectorData = csv.reader(csvfile, delimiter=',')
	next(vectorData, None) #Skips the header
	for kpt in vectorData:
		dots.append( [float(kpt[0]), float(kpt[1]), float(kpt[2]), float(kpt[3]), float(kpt[5]), float(kpt[6])] )

X = dots

print(X)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
