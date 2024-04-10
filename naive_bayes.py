## Imports and Setup
from nltk.corpus import stopwords
import numpy as np
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re

## Helper functions
def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    #tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
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

def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    #for y, tweet in zip([y[0] for y in ys], tweets):
    for y, tweet in zip(ys, tweets):    
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1
            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    return result

## Naive Bayes Class for Tweet Sentiment Analysis
class SA_NaiveBayes():

    def __init__(self, freqs):
        self.freqs = freqs
        self.loglikelihood = {}
        self.logprior = 0
    
    def train(self, x, y):
        '''
        Input:
            x: a list of tweets
            y: a list of labels corresponding to the tweets (0,1)
        Output:
            logprior: the log prior.
            loglikelihood: the log likelihood of you Naive bayes equation.
        '''
        # calculate V, the number of unique words in the vocabulary
        vocab = [word for word, s in self.freqs.keys()]
        V = len(list(set(vocab)) ) 

        # calculate N_pos, N_neg, V_pos, V_neg
        N_pos = sum([self.freqs[word, s] if s == 1 else 0 for word, s in self.freqs.keys()])
        N_neg = sum([self.freqs[word, s] if s == 0 else 0 for word, s in self.freqs.keys()])
        
        # Calculate D, the number of documents
        D = len(x)
        # Calculate D_pos, the number of positive documents
        D_pos = sum(y == 1)
        # Calculate D_neg, the number of negative documents
        D_neg = sum(y == 0)

        # Calculate logprior
        self.logprior = np.log(D_pos) - np.log(D_neg)
        
        # For each word in the vocabulary...
        for word in vocab:
            # get the positive and negative frequency of the word
            freq_pos = self.freqs[(word, 1)] if (word, 1) in self.freqs else 0
            freq_neg = self.freqs[(word, 0)] if (word, 0) in self.freqs else 0
            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)
            # calculate the log likelihood of the word
            self.loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)
    
    def parameters(self):
        return self.logprior, self.loglikelihood

    def predict(self, tweet):
        '''
        Input:
            tweet: a string
        Output:
            p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

        '''
        # process the tweet to get a list of words
        word_l = process_tweet(tweet)
        # initialize probability to zero
        p = 0
        # add the logprior
        p += self.logprior

        for word in word_l:
            # check if the word exists in the loglikelihood dictionary
            if word in self.loglikelihood:
                # add the log likelihood of that word to the probability
                p += self.loglikelihood[word]

        return p
    
    def get_ratio(self, word):
        '''
        Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
            Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
        '''
        pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
        # use lookup() to find positive counts for the word (denoted by the integer 1)
        pos_neg_ratio['positive'] = self.freqs[(word, 1)] if (word, 1) in self.freqs else 0
        # use lookup() to find negative counts for the word (denoted by integer 0)
        pos_neg_ratio['negative'] = self.freqs[(word, 0)] if (word, 0) in self.freqs else 0
        # calculate the ratio of positive to negative counts for the word
        pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
        
        return pos_neg_ratio
    
    def get_words_by_threshold(self, label, threshold):
        '''
        Input:
            label: 1 for positive, 0 for negative
            threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
        Output:
            word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
            example of a key value pair:
            {'happi':
                {'positive': 10, 'negative': 20, 'ratio': 0.5}
            }
        '''
        word_list = {}

        for key in self.freqs.keys():
            word, _ = key[0], key[1]
            # get the positive/negative ratio for a word
            pos_neg_ratio = self.get_ratio(word)

            # if the label is 1 and the ratio is greater than or equal to the threshold...
            if label == 1 and pos_neg_ratio['ratio'] >= threshold:
                # Add the pos_neg_ratio to the dictionary
                word_list[word] = pos_neg_ratio
            # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
            elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
                # Add the pos_neg_ratio to the dictionary
                word_list[word] = pos_neg_ratio
            # otherwise, do not include this word in the list (do nothing)
        
        return word_list
    
    def eval(self, x, y):
        """
        Input:
            x: A list of tweets
            y: the corresponding labels for the list of tweets
        Output:
            accuracy: (# of tweets classified correctly)/(total # of tweets)
        """
        accuracy = 0  # return this properly
        y_hats = []
        for tweet in x:
            # if the prediction is > 0
            if self.predict(tweet) > 0:
                # the predicted class is 1
                y_hat_i = 1
            else:
                # otherwise the predicted class is 0
                y_hat_i = 0
            # append the predicted class to the list y_hats
            y_hats.append(y_hat_i)

        # error is the average of the absolute values of the differences between y_hats and test_y
        error = sum(y_hats != y) / len(y_hats)
        # Accuracy is 1 minus the error
        accuracy = 1 - error

        return accuracy