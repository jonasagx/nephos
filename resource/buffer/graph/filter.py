from cv2 import norm
from numpy import mean
from math import atan2
from scipy.spatial import distance

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
            distances_sum += norm(p1, p2)

        mean = distances_sum/len(track['matches'])
        #print mean

        for match in track['matches']:
            p1 = track['kp1'][match.queryIdx].pt
            p2 = track['kp2'][match.trainIdx].pt
            #print str(norm(p1, p2)) + ", ",
            if norm(p1, p2) < mean:
                filtered.append((p1, p2))

    return filtered

def basicFormater(trackSerie):
    filtered = []

    for key, track in trackSerie.items():

            for match in track['matches']:
                p1 = track['kp1'][match.queryIdx].pt
                p2 = track['kp2'][match.trainIdx].pt
                angle = atan2(p1[0] - p2[0], p1[1] - p2[1])
                filtered.append((p1, p2, distance.euclidean(p1, p2), angle, key))

    return filtered