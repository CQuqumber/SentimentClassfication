from efficiency import *
from preview_label import *

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
mlp.train(reviews[:-1000],labels[:-1000])
mlp.test(reviews[-1000:],labels[-1000:])