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
		for review in reviews:
			for word in review.split(" "):
				review_vocab.add(word)
		self.review_vocab = list(review_vocab)


		label_vocab = set()
		for label in labels:
			label_vocab.add(label)
		self.label_vocab = list(label_vocab)

		self.review_vocab_size = len(self.review_vocab)
		self.label_vocab_size = len(self.label_vocab)


		self.word2index = {}
		for i, word in enumerate(self.review_vocab):
			self.word2index[word] = i

		self.label2index = {}
		for i, label in enumerate(self.label_vocab):
			self.label2index[label] = i

	def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# set number of nodes in input, hidden and output layers
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		#Initialze weights
		self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

		self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, \
											(self.hidden_nodes, self.output_nodes) )

		self.learning_rate = learning_rate

		self.layer_0 = np.zeros((1, input_nodes))























