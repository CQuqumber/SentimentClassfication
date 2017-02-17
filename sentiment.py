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
		self.layer_1 = np.zeros((1, hidden_nodes))

	def update_input_layer(self, review):
		# clear out previous state, reset the layer to be all 0s
		self.layer_0 *= 0
		for word in review.split(" "):
			#if(word in self.word2index.key()):	# Redundant
				#self.layer_0[0][self.word2index[word]] += 1
			self.layer_0[0][self.word2index[word]] = 1	#  Noise reduction

	def get_target_for_label(self, label):
		if(label == 'POSITIVE'):
			return 0
		else:
			return 1


	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def sigmoid_output_2_derivative(self, output):
		return output * (1 - output)


	def train(self, training_reviews_raw, training_labels):
		
		training_reviews = list()
		for review in training_reviews_raw:
			indices = set()
			for word in review.split(" "):
				if(word in self.word2index.keys()):
					indices.add(self.word2index[word])
			training_reviews.append(list(indices))

		#insert debugging assertions
		assert(len(training_reviews) == len(training_labels))

		correct_so_far = 0

		start = time.time()

		for i in range(len(training_reviews)):

			review = training_reviews[i]
			label = training_labels[i]

			# Forward pass

			# Input Layer
			#self.update_input_layer(review)

			# Hidden layer
			#layer_1 = self.layer_0.dot(self.weights_0_1)
			self.layer_1 *= 0
			for index in review:
				self.layer_1 += self.weights_1_2[index]

			# Output layer
			layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

			### Backward pass
			## Output error
			layer_2_error = layer_2 - self.get_target_for_label(label)
			#  O/p error difference between desire and actual
			layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

			## Baclpropagated error
			layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
			# error propagated to the hidden layer
			layer_1_delta = layer_1_error
			# hidden layer gradient - no nonlinearity so it's the same as the error

			## Update the weights
			self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
			# update hidden - to - output weight with gradient descent step

			#self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate
			# update the input 2 hidden weights with gradient descent step

			for index in review:
				self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate

			if(np.abs(layer_2_error) < 0.5):
				correct_so_far += 1

			reviews_per_second = i / float(time.time() - start)	# start = time.time()

			sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] +\
				"% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + \
			 	" #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + \
			 	" Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            
			if(i % 2500 == 0):
				print("")

	def test(self, testing_reviews, testing_labels):

		correct = 0

		start = time.time()

		for i in range(len(testing_reviews)):
			pred = self.run(testing_reviews[i])
			if(pred == testing_labels[i]):
				correct += 1

			reviews_per_second = i / float(time.time() - start)

			sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] + \
			"% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + \
			"% #Correct:" + str(correct) + " #Tested:" + str(i+1) + \
			" Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

	def run(self, review):

		# Input Layer
		#self.update_input_layer(review.lower())

		# Hidden Layer
		self.layer_1 *= 0
		unique_indices = set()
		for word in review.lower().split(" "):
			if word in self.word2index.key():
				unique_indices.add(self.word2index[word])
		for index in unique_indices:
			self.layer_1 += self.weights_0_1[index]

		# Output layer
		layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

		if(layer_2[0] > 0.5):
			return 'POSITIVE'
		else:
			return 'NEGATIVE'

