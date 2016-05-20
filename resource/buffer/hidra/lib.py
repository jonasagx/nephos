import os
import cv2 as cv

matcher = cv.BFMatcher(crossCheck=True)
detector = cv.xfeatures2d.SURF_create()

def loadPictures(path):
	assert path.find('/') != -1, "Pass a valid path with slash(/)"
	l = os.listdir(path)
	l.sort()
	return l

def readPictures(pics, path):
	return [{"index":index, "pic": cv.imread(path + pic)} for index, pic in enumerate(pics)]

def dumpKey(key):
	return {
		"pt": key.pt,
        "size": key.size,
        "angle": key.angle,
        "response": key.response,
        "octave": key.octave, 
        "class_id": key.class_id
        }

def detect(img):
	keys, des = detector.detectAndCompute(img['pic'], None)
	return {"keys": [ dumpKey(key) for key in keys], "index": img['index']}

def match(a, b):
	return matcher.match(a, b)

def basicFormater(trackSerie):
    filtered = []

    for key, track in trackSerie.items():
            for match in track['matches']:
                p1 = track['kp1'][match.queryIdx].pt
                p2 = track['kp2'][match.trainIdx].pt
                angle = atan2(p1[0] - p2[0], p1[1] - p2[1])
                filtered.append((p1, p2, distance.euclidean(p1, p2), angle, key))

    return filtered