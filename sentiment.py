from preview_label import *
import time
import sys
import numpy as np


class SentimentNetwork:
	def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
		# set our random number generator
		np.random.seed(1)

		self.pre_process_data(reviews, labels)

		self.init_network(len(self.review_vocab), hidden_nodes, 1 ,learning_rate)

	def pre_process_data(self, reviews, labels):
		review_vocab = set()