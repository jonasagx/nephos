#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv

class Photo:
	"""Represents a photo"""
	def __init__(self, filename):
		self.filename = filename

	def load(self, mode=2):
		self.im = cv.imread(self.filename, 0)

	def filter(self, max, min):
		print self.im.shape
		x, y = self.im.shape
		for i in range(x):
			for j in range(y):
				if self.im.item(i, j) > min and self.im.item(i, j) < max:
					pass
				else:
					self.im.itemset(i, j, 0)
    
