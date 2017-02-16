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

	def get_target_for_label(self, label):
		if(label == 'POSITIVE'):
			return 0
		else:
			return 1


	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def sigmoid_output_2_derivative(self, output):
		return output * (1 - output)


	def train(self, training_reviews, training_labels):
		#insert debugging assertions
		assert(len(training_reviews) == len(training_labels))

		correct_so_far = 0

		start = time.time()

		for i in range(len(training_reviews)):

			review = training_reviews[i]
			label = training_labels[i]

			# Forward pass

			# Input Layer
			self.update_input_layer(review)

			# Hidden layer
			layer_1 = self.layer_0.dot(self.weights_0_1)

			# Output layer
			layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

			### Backward pass
			layer_2_error = layer_2 - self.get_target_for_label(label)
			#  O/p error difference between desire and actual

			layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)



















