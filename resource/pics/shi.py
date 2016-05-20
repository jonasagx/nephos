import numpy as np
import cv2
#from matplotlib import pyplot as plt
import sys

assert len(sys.argv) > 1, "Pass an image as param"
filename = sys.argv[1]
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

#plt.imshow(img),plt.show()
cv2.imshow("Shi", img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
