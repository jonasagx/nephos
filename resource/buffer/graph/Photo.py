#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv

class Photo:
	"""docstring for Photo"""
	def __init__(self, filename):
		self.filename = filename

	def load(self, mode=2):
		self.im = cv.imread(self.filename)
