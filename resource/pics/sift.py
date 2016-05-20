import numpy as np
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")

import cv2

img = cv2.imread(sys.argv[1])
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
kp = surf.detect(gray,None)

print len(kp)

img2 = np.zeros((1, 1, 1), np.uint8)
img=cv2.drawKeypoints(gray, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=200)

cv2.imwrite('sift_keypoints.jpg',img)
