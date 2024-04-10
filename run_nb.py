## Dependencies
from naive_bayes import SA_NaiveBayes, count_tweets
import nltk
import numpy as np
from nltk.corpus import twitter_samples
from datetime import datetime
print(str(datetime.now()) + ': ' + 'Dependencies loaded')

## Load Dataset
nltk.download('twitter_samples')
nltk.download('stopwords')
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
print(str(datetime.now()) + ': ' + 'Dataset loaded')

## Split Data into Training and Test
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
# combine positive and negative labels
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))
print(str(datetime.now()) + ': ' + 'Dataset split')

## Build the freqs dictionary
freqs = count_tweets({}, train_x, train_y)
print(str(datetime.now()) + ': ' + 'Freqs dictionary created')

## Instantiate and Train Model
model = SA_NaiveBayes(freqs)
print(str(datetime.now()) + ': ' + 'Model instantiated')
# Train
model.train(train_x, train_y)
print(str(datetime.now()) + ': ' + 'Model trained')

## Evaluate Model
# Training Set
train_acc = model.eval(train_x, train_y)
print(f"Training accuracy: {train_acc:.4f}")
# Test Set
test_acc = model.eval(test_x, test_y)
print(f"Test accuracy: {test_acc:.4f}")
print(str(datetime.now()) + ': ' + 'Model accuracy evaluation completed')

## Predict Individual Examples
for tweet in ['She smiled.', 'I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print(f'{tweet}: {model.predict(tweet)}')
print(str(datetime.now()) + ': ' + 'Model predictions completed')

## Get ratio of word
model.get_ratio('happi')
## Get words by threshold
# Negative sentiment
print(model.get_words_by_threshold(label=0, threshold=0.05))
# Positive sentiment
print(model.get_words_by_threshold(label=1, threshold=10))
print(str(datetime.now()) + ': ' + 'Ratio & words by treshold functions executed')