from more_efficiency import *
from preview_label import *

mlp = SentimentNetwork(reviews[:-4000],labels[:-4000], learning_rate=0.01)
mlp.train(reviews[:-4000],labels[:-4000])
mlp.test(reviews[-4000:],labels[-4000:])