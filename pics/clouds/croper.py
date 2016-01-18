#!/usr/bin/python3.4
from PIL import Image
import os
import sys

rootPath = sys.argv[1]
savePath = sys.argv[2]

fileList = os.listdir(rootPath)
s = [309, 334, 501, 522]

for f in fileList:
    if f.find("jpg") >= 0:
        im = Image.open(rootPath + "/" + f)
        print(f)
        im.crop(s).save( savePath + "/" + im.filename.replace(".jpg", "_square.jpg"))
