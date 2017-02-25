import time
import sys
import numpy as np
from preview_label import *


class SentimentNetwork:
	def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.):
		np.random.seed(1)

		self.pre_process_data(reviews)

		self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)


	def pre_process_data(self, reviews):

		review_vocab = set()
		for review in reviews:
			for word in review.split(" "):
				review_vocab.add(word)
		self.review_vocab = list(review_vocab)


		label_vacob = set()

		for label in labels:
			label_vacob.add(label)

		self.label_vacob = list(label_vacob)

		self.label_vacob_size = len(self.vacob_vocab)
		self.review_vocab_size = len(self.review_vocab)


		self.word2index = {}
		for i, word in enumerate(self.review_vocab):
			self.word2index[word] = i

		self.label2index = {}
		for i, label in enumerate(self.label_vacob):
			self.label2index[label] = i






















































































