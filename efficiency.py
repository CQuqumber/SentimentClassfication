import time
import sys
import numpy as np
from preview_label import *

class SentimentNetwork:
    """docstring for ClassName"""
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = .1):

        np.random.seed(1)

        self.pre_process_data(reviews, labels)

        self.init_network(len(self.review_vocab), hidden_nodes, 1 , learning_rate)

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


        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5,
                                                    (self.hidden_nodes, self.output_nodes))

        self.learning_rate = learning_rate

        self.layer_0 = np.zeros((1, input_nodes))




    def update_input_layer(self, review):
        self.layer_0 *= 0
        for word in review.split(" "):
            self.layer_0[0][self.word2index[word]] = 1



    def get_target_for_label(self, label):
        if (label == "POSITIVE"):
            return 1
        else:
            return 0



    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)


    def train(self, training_reviews, training_labels):
        assert(len(training_reviews) == len(training_labels))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]


            self.update_input_layer(review)

            layer_1 = self.layer_0.dot(self.weights_0_1)

            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))


            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)


            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error

            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate

            if(np.abs(layer_2_error) < .5):
                correct_so_far += 1

            reviews_per_second = i /float(time.time() - start)

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")



            if(i % 2000 == 0):
                print("")



    def run(self, review):

        self.update_input_layer(review)

        layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        if(layer_2[0] > .5):
            return "POSITIVE"
        else:
            return "NEGATIVE"


    def test(self, training_reviews, training_labels):

        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == training_labels[i]):
                correct += 1

            review_per_second = i / float(time.time() - start)
            sys.stdout.write("\rProcess:" , str(100 *i / float(len(training_reviews)))[:4]\
                                + "% Speed(reviews/sec): " + str(review_per_second)[0:5]\
                                + " # Correct:" + str(correct_so_far) \
                                + " # Trained:" + str(i + 1) \
                                + " Training Accuracy:" \
                                + str(correct_so_far * 100) / float(i + 1)[:4] + "%" )








