import time
import sys
import numpy as np
from preview_label import *

class SentimentNetwork:
	"""docstring for SentimentNetwork"""
	def __init__(self, reviews, labels min_count = 10, polarity_cutoff = 0.1, hidden_nodes = 10, learning_rate = 0.1):
		
		np.random.seed(1)

		self.pre_process_data(reviews, polarity_cutoff, min_count)
		
		self.init_network(len(self.reviews_vocab), hidden_nodes, 1, learning_rate)


	def pre_process_data(self, reviews, polarity_cutoff, min_count):
		
		positive_counts = Counter()
		negative_counts = Counter()
		total_counts = Counter()

		for i in range(len(reviews)):
			if(labels[i] == 'POSITIVE'):
				for word in reviews[i].split(" "):
					positive_counts[word] += 1
					total_counts[word] += 1
			else:
				for word in reviews[i].split(" "):
					negative_counts[word] += 1
					total_counts[word] += 1

		pos_neg_ratios = Counter()



		for term, cnt in list(total_counts.most_common()):
			if (cnt > 200):
				pos_neg_ratio = positive_counts[term] / float(1 + negative_counts[term])
				pos_neg_ratios[term] = pos_neg_ratio


		for word, ratio in pos_neg_ratios.most_common():
			if(ratio > 1):
				pos_neg_ratios[word] = np.log(ratio)
			else:
				pos_neg_ratios[word] = np.log(0.01 + ratio)


		for word in review.split(" "):
			if(total_counts[word] > min_count):
				if(word in pos_neg_ratios):
					if(pos_neg_ratios[word] >= polarity_cutoff or pos_neg_ratios[word] <= -polarity_cutoff):
						reviews_vocab.add(word)
				else:
					reviews_vocab.add(word)
		self.reviews_vocab = list(reviews_vocab)


		label_vocab = set()
		for label in labels:
			label_vocab.add(label)
		self.label_vocab = list(label_vocab)

		self.reviews_vocab_size = len(self.reviews_vocab)
		self.label_vocab_size = len(self.label_vocab)

		self.word2index = {}
		for i, word in enumerate(self.reviews_vocab):
			self.word2index[word] = i

		self.label2index = {}
		for i, label in enumerate(self.label_vocab):
			self.label2index[label] = i



	def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
		self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
													(self.hidden_nodes, self.output_nodes))

		self.learning_rate = learning_rate

		self.layer_0 = np.zeros((1, input_nodes))
		self.layer_1 = np.zeros((1, hidden_nodes))


	def sigmoid(self, x):
		return 1 /( 1 + np.exp(-x))


	def sigmoid_output_2_derivative(self, output):
		return output * ( 1 - output)


	def update_input_layer(self, review):
		self.layer_0 *= 0
		for word in review.split(" "):
			self.layer_0[0][word2index[word]] = 1


	def get_target_for_label(label):
		if(labe == "POSITIVE"):
			return 1
		else:
			return 0


	def train(self, training_review_raw, training_labels):
		
		training_reviews = list()
		for review in training_review_raw:
			indices = set()
			for word in review.split(" "):
				indices.add(self.word2index[word])
		training_reviews.append(list(indices))


		assert(len(training_reviews) == len(training_labels))

		correct_so_far = 0

		start = time.time()

		for i in range(len(training_reviews)):

			review = training_reviews[i]
			label = training_labels[i]


			self.layer_1 *= 0
			for index in review:
				self.layer_1 += self.weights_0_1[index]


			layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))


			layer_2_error = layer_2 - self.get_target_for_label(label)
			layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)


			layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
			layer_1_delta = layer_1_error

			self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate


			for index in review:
				self.weights_0_1[index] -= layer_1_delta * self.learning_rate


			if(layer_2 >= 0.5 and label == 'POSITIVE'):
				correct_so_far += 1

			if(layer_2 < 0.5 and label == 'NEGATIVE'):
				correct_so_far += 1

			reviews_per_second = i / float(time.time() - start)
			sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
           		+ "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
           		+ " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
            	+ " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")







	def run(self, review):
		
		self.layer_1 *= 0

		unique_indices = set()

		for word in review.split(" "):
			unique_indices.add(word)

		for index in unique_indices:
			self.layer_1 += self.weights_0_1[index]

		layer_2 = self.sigmoid_output_2_derivative(self.layer_1.dot(self.weights_1_2))

		if(layer_2[0] >= 0.5 ):
			return 'POSITIVE'
		else:
			return 'NEGATIVE'


	def test(self, training_reviews, training_labels):
		correct = 0

		start = time.time()

		for i in range(len(training_reviews)):
			pred = self.run(training_reviews[i])
			if(pred == training_labels[i]):
				correct += 1

			reviews_per_second = i /float(time.time() - start)

			sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
            	+ "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
            	+ " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
            	+ " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")



























































