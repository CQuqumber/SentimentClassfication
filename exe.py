from sentiment import *
from preview_label import *

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)

# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])

# train the network
mlp.train(reviews[:-1000],labels[:-1000])