## Imports and Setup
import numpy as np

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

## Helper functions
def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))

    return h

## Logistic Regression Class for Tweet Sentiment Analysis)
class SA_LogisticRegression():

    def __init__(self, freqs):
        self.freqs = freqs
        self.theta = np.zeros((3, 1))
    
    def single_encode(self, tweet):
        '''
        Input:
            tweet: a string containing one tweet
        Output:
            x: a feature vector of dimension (1,3)
        '''
        # process_tweet tokenizes, stems, and removes stopwords
        word_l = process_tweet(tweet)

        # 3 elements for [bias, positive, negative] counts
        x = np.zeros(3)

        # bias term is set to 1
        x[0] = 1

        # loop through each word in the list of words
        for word in word_l:

            # increment the word count for the positive label 1
            x[1] += self.freqs[(word, 1)] if (word, 1) in self.freqs else 0

            # increment the word count for the negative label 0
            x[2] += self.freqs[(word, 0)] if (word, 0) in self.freqs else 0

        x = x[None, :]  # adding batch dimension for further processing
        assert(x.shape == (1, 3))
        return x
    
    def encode(self, tweets):
        '''
        Input:
            tweets: a  list containing one tweet per element
        Output:
            x: a feature matrix of dimension (m,3)
        '''
        x = np.zeros((len(tweets), 3))
        for i in range(len(tweets)):
            x[i, :]= self.single_encode(tweets[i])
        return x

    def train(self, x, y, alpha=1e-9, num_iters=1500, verbose=False, verb_interval=100):
        '''
        Input:
            x: matrix of features which is (m,n+1)
            y: corresponding labels of the input matrix x, dimensions (m,1)
            theta: weight vector of dimension (n+1,1)
            alpha: learning rate
            num_iters: number of iterations you want to train your model for
            verbose: boolean corresponding to whether the model should print the cost function during training
            verb_interval: number of iterations after which the model prints the cost function when verbose is True
        Output:
            J: the final cost
            theta: your final weight vector
        Hint: you might want to print the cost to make sure that it is going down.
        '''

        # get 'm', the number of rows in matrix x
        m = x.shape[0]

        for i in range(0, num_iters):

            # get z, the dot product of x and theta
            z = np.dot(x, self.theta)

            # get the sigmoid of z
            h = sigmoid(z)

            # calculate the cost function
            J = -(1/m)*(np.dot(y.T, np.log(h)) + np.dot((1-y.T), np.log(1 - h)))

            # update the weights theta
            self.theta -= (alpha/m)*(np.dot(x.T, (h - y)))

            # Print loss
            if (verbose) and (i % verb_interval == 0):
                print(f"Step {i}: Total cost {float(J):.4f}")

        J = float(J)
        # Print loss
        if verbose:
            print(f"Final cost: {J:.4f}")
    
    def parameters(self):
        return self.theta
    
    def predict(self, tweet):
        '''
        Input:
            tweet: a string
        Output:
            y_pred: the probability of a tweet being positive or negative
        '''

        # extract the features of the tweet and store it into x
        x = self.single_encode(tweet)
        
        # make the prediction using x and theta
        y_pred = sigmoid(np.dot(x, self.theta))

        return y_pred
    
    def eval(self, x, y):
        """
        Input:
            x: a list of tweets
            y: (m, 1) vector with the corresponding labels for the list of tweets
        Output:
            accuracy: (# of tweets classified correctly) / (total # of tweets)
        """

        # the list for storing predictions
        y_hat = []

        for tweet in x:
            # get the label prediction for the tweet
            y_pred = self.predict(tweet)

            if y_pred > 0.5:
                # append 1.0 to the list
                y_hat.append(1.0)
            else:
                # append 0 to the list
                y_hat.append(0.0)

        # With the above implementation, y_hat is a list, but test_y is (m,1) array
        # convert both to one-dimensional arrays in order to compare them using the '==' operator
        accuracy = np.sum(np.array(y_hat).reshape(len(y_hat), 1) == y)/(y.shape[0])

        return accuracy