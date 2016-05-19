import numpy as np
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")

import cv2
#from matplotlib import pyplot as plt

assert len(sys.argv) > 2, "Pass two images as params"
f1 = sys.argv[1]
f2 = sys.argv[2]

img1 = cv2.imread(f1, 0)          # queryImage
img2 = cv2.imread(f2, 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIRF_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)

# Apply ratio test
'''good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
'''

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)

#plt.imshow(img3),plt.show()
cv2.imshow("Match", img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
