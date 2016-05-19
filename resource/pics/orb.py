import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
import numpy as np
import cv2

img1 = cv2.imread(sys.argv[1]) # Original image
img2 = cv2.imread(sys.argv[2]) # Rotated image

# Create ORB detector with 1000 keypoints with a scaling pyramid factor
# of 1.2
#orb = cv2.ORB_create(1000, 1.2)

surf = cv.xfeatures2d.SURF_create()
# Detect keypoints of original image
(kp1,des1) = surf.detectAndCompute(img1, None)

# Detect keypoints of rotated image
(kp2,des2) = surf.detectAndCompute(img2, None)

# Create matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Do matching
matches = bf.match(des1,des2)

# Sort the matches based on distance.  Least distance
# is better
#matches = sorted(matches, key=lambda val: val.distance)

# Show only the top 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

cv2.imshow("Match", img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

