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
		point = (int(kpt[0]), int(kpt[1]), int(kpt[2]), int(kpt[3]), float(kpt[5]))
		print distance.euclidean(zero, point)
		print distance.mahalanobis(zero, point)
		# dots.append( (int(kpt[0]), int(kpt[1]), int(kpt[2]), int(kpt[3]), float(kpt[5])) )
		# dots.append( (int(kpt[0]), int(kpt[1])) )

# X = np.array(dots)

# print(X)

# db = DBSCAN(eps=0.5, min_samples=10, metric='euclidean', algorithm='auto', leaf_size=2, p=None, random_state=None).fit_predict(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# print(db)

# Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
