from datetime import datetime
from util import loadFiles
import sys
import time

def validSerie(filesList):
	timePatter = '%Y%m%d%H%M'
	filesListSize = len(filesList)

	startIndex = 0
	maxSize = 0
	lastMaxSize = 0
	lastStartIndex = 0

	for index in range(filesListSize - 2):
		img1 = filesList[index].split("_")[1]
		img2 = filesList[index + 1].split("_")[1]

		t1 = datetime.strptime(img1, timePatter)
		t2 = datetime.strptime(img2, timePatter)

		unixTime1 = time.mktime(t1.timetuple())
		unixTime2 = time.mktime(t2.timetuple())

		if unixTime1 > unixTime2:
			raise ValueError("List of files is not sorted")

		dt = unixTime2 - unixTime1
		if dt > 30 * 60:
			if maxSize > lastMaxSize:
				lastMaxSize = maxSize
				lastStartIndex = startIndex

			startIndex = index + 1
			maxSize = 0
		else:
			maxSize += 1

	files = []
	for index in range(lastStartIndex, lastStartIndex + lastMaxSize + 1):
		files.append(filesList[index])
	return files

path = sys.argv[1]
filesList = loadFiles(path)

print validSerie(filesList)