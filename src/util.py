import os
import time
import cv2 as cv
from numpy import mean
from math import atan2
from numpy import mean
from scipy.spatial import distance

def loadFiles(path):
	assert len(path) > 0, "Pass folder path as param"
	if not path.endswith("/"):
		path += "/"
	filesList = os.listdir(path)
	filesList.sort()
	return filesList

def filterDistances(trackSerie):
    filtered = []
    for key, track in trackSerie.items():
        #Get list of distances
        m = mean( [track['matches'][i].distance for i in xrange(len(track['matches']))] )
        #print mean

        for match in track['matches']:
            if match.distance < m * 0.5:
                #distances.append(match.distance)
                filtered.append( (track['kp1'][match.queryIdx].pt, track['kp2'][match.trainIdx].pt) )
                #print track['kp1'][match.queryIdx].pt, track['kp2'][match.trainIdx].pt

    return filtered

def filterEuDistances(trackSerie):
    filtered = []
    for key, track in trackSerie.items():
        #Get list of distances
        distances_sum = 0.0

        for match in track['matches']:
            p1 = track['kp1'][match.queryIdx].pt
            p2 = track['kp2'][match.trainIdx].pt
            distances_sum += cv.norm(p1, p2)

        mean = distances_sum/len(track['matches'])
        #print mean

        for match in track['matches']:
            p1 = track['kp1'][match.queryIdx].pt
            p2 = track['kp2'][match.trainIdx].pt
            #print str(norm(p1, p2)) + ", ",
            if cv.norm(p1, p2) < mean:
                filtered.append((p1, p2))

    return filtered

def basicFormater(trackSerie):
    lines = []
    for key, track in trackSerie.items():
            filtered = []
            for match in track['matches']:
                p1 = track['kp1'][match.queryIdx].pt
                p2 = track['kp2'][match.trainIdx].pt
                angle = atan2(p1[0] - p2[0], p1[1] - p2[1])
                filtered.append((p1, p2, distance.euclidean(p1, p2), angle, key))
            lines.append(filtered)
    return lines

def printer(photos):
    s = "x1, y1, x2, y2, d, a, t\n"
    for lines in photos:
        for line in lines:
            s += "%d, %d, %d, %d, %.3f, %.3f, %d\n" % (line[0][0], line[0][1], line[1][0], line[1][1], line[2], line[3], line[4])
    return s

def countPositives(photos):
    results = []
    for lines in photos:
        m = mean([d[2] for d in lines])
        positives = [d[2] <= 2 * m for d in lines].count(True)
        results.append(positives)
    return results

def globalPrinter(serie, filesList, path):
    kps = countKeypointsByPhoto(serie)
    pt = processingTimeByPhoto(serie)
    stds = stdByPhoto(filesList, path)
    means = meanByPhoto(filesList, path)

    s = "kpts, pt, std, mean\n"
    for photo in range(len(filesList)):
        s += "%d, %.5f, %.3f, %.3f\n" % (kps[photo], pt[photo], stds[photo], means[photo])
    return s

def matchPrinter(flatTracks):
    tracks = countMatchesByPair(flatTracks)
    positives = countPositives(flatTracks)
    s = "nm, pm\n"
    for i in range(len(tracks)):
        s += "%d, %d\n" % (tracks[i], positives[i])
    return s

def countKeypointsByPhoto(serie):
    counter = []
    last = []
    for key, track in serie.items():
        counter.append(len(track['kp1']))
        last = len(track['kp2'])
    counter.append(last)
    return counter

def processingTimeByPhoto(serie):
    time = []
    last = []
    for key, track in serie.items():
        time.append(track['pt1'])
        last = track['pt2']
    time.append(last)
    return time

def countMatchesByPair(flatTracks):
    matches = []
    for track in flatTracks:
        matches.append( len(track) )
    return matches

def stdByPhoto(filesList, path):
    stats = []
    for file in filesList:
        im = cv.imread(path + file)
        stats.append(im.std())
    return stats

def meanByPhoto(filesList, path):
    stats = []
    for file in filesList:
        im = cv.imread(path + file)
        stats.append(im.mean())
    return stats

def seekMatches(detector, matcher, filesList, path):
    trackSerie = {}
    for index in xrange(len(filesList) - 1):
        img1 = cv.imread(path + filesList[index], 0)
        img2 = cv.imread(path + filesList[index + 1], 0)

        # if(img1 == None or img2 == None):
            # continue
        t11 = time.time()
        (kp1, des1) = detector.detectAndCompute(img1, None)
        t12 = time.time()

        t21 = time.time()
        (kp2, des2) = detector.detectAndCompute(img2, None)
        t22 = time.time()

        if(des1 == None or des2 == None):
            trackSerie[index] = {'matches': [],
                              'kp1': [],
                              'kp2': [],
                              'des1': [],
                              'des2': [],
                              'pt1': t12 - t11,
                              'pt2': t22 - t21,
                              'tm': tm2 - tm1
                            }            
            continue

        tm1 = time.time()
        matches = matcher.match(des1, des2)
        tm2 = time.time()

        trackSerie[index] = {'matches': matches,
                              'kp1': kp1,
                              'kp2': kp2,
                              'des1': des1,
                              'des2': des2,
                              'pt1': t12 - t11,
                              'pt2': t22 - t21,
                              'tm': tm2 - tm1
                            }
    return trackSerie
