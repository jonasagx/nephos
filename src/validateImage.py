import os
import sys

from PIL import Image

def loadFiles(path, fileExtention):
	assert len(path) > 0, "Pass folder path as param"
	if not path.endswith("/"):
		path += "/"
	filesList = os.listdir(path)
	
	for file in filesList:
		if not file.endswith(fileExtention):
			filesList.remove(file)
	
	filesList.sort()
	return filesList

path = sys.argv[1]
filesList = loadFiles(path, ".jpg")
cropSquare = [79, 109, 509, 346]

for filename in filesList:
	print(filename)
	im = Image.open(path + filename)
	im.crop(cropSquare).save("/tmp/blabla.jpg")
	im.close()