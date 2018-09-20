import math
import random

class Random_Classifier(object):

	"""Randomly classify wine from 1 to 10"""

	def __init__(self):
		self.reset()

	def reset(self):
		None

	def classify(self, row):
		return random.randint(1, 10)		