## Dependencies
from basic_lr import SA_LogisticRegression, build_freqs
import nltk
import numpy as np
from nltk.corpus import twitter_samples
from datetime import datetime
print(str(datetime.now()) + ': ' + 'Dependencies loaded')

## Hyperparameters
alpha=1e-9
num_iters=4500
verbose=True
verb_interval=100
print(str(datetime.now()) + ': ' + 'Hyperparameters loaded')

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
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
print(str(datetime.now()) + ': ' + 'Dataset split')

## Build Frequency Dictionary
freqs = build_freqs(train_x, train_y)
print(str(datetime.now()) + ': ' + 'Freqs dictionary created')

## Instantiate and Train Model
model = SA_LogisticRegression(freqs)
print(str(datetime.now()) + ': ' + 'Model instantiated')
# Encode
train_encX = model.encode(train_x)
print(str(datetime.now()) + ': ' + 'Training data encoded')
# Train
model.train(train_encX, train_y, alpha, num_iters, verbose, verb_interval)
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
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print(f'{tweet}: {model.predict(tweet)[0]}')
print(str(datetime.now()) + ': ' + 'Model predictions completed')