from cv2 import norm
from numpy import mean

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
            if norm(p1, p2) < mean * 0.2:
                filtered.append((p1, p2))    

    return filtered
