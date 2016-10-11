# make_corpus.py

import pickle
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt 

# load processed lines
with open('clinton.pickle', 'rb') as f : clinton = pickle.load(f)
with open('trump.pickle', 'rb') as f : trump = pickle.load(f)
# with open('moderator.pickle', 'rb') as f : moderator = pickle.load(f)
# with open('dems.pickle', 'rb') as f : dems = pickle.load(f)
# with open('reps.pickle', 'rb') as f : reps = pickle.load(f)

# set up data frame to store lines and labels
# note labels are: 0 to 4 in order listed below (order added to all_lines)
all_clinton = list(itertools.chain.from_iterable(clinton))
all_trump = list(itertools.chain.from_iterable(trump))
# all_moderator = list(itertools.chain.from_iterable(moderator))
# all_dems = list(itertools.chain.from_iterable(dems))
# all_reps = list(itertools.chain.from_iterable(reps))
all_lines = all_clinton + all_trump #+ all_moderator + all_dems + all_reps
# labels = np.repeat(np.arange(5), [len(all_clinton), len(all_trump), len(all_moderator), len(all_dems), len(all_reps)])
labels = np.repeat(np.arange(2), [len(all_clinton), len(all_trump)])#, len(all_moderator), len(all_dems), len(all_reps)])
lines_df = pd.DataFrame({'lines': all_lines, 'labels': labels})

# split data to train, cross validation, test sets
# 60/20/20 splits
lines_df = pd.DataFrame(np.random.permutation(lines_df)) # shuffle dataframe
lines_df.columns = ['labels', 'lines']
slice_size = int(lines_df.shape[0]*0.2)
train_dat = lines_df[0 : 3*slice_size]
xval_dat = lines_df[3*slice_size : 4*slice_size]
test_dat = lines_df[4*slice_size :]

# create feature extractor object
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')

# create vocabulary and feature list based on first dems collection of lines
X_train = vectorizer.fit_transform(train_dat['lines'])
y_train = [int(x) for x in train_dat['labels']]
# X_train.toarray() # see features matrix count (sparse matrix)

# ways to see list of words used as features
# tom = vectorizer.get_feature_names()
# tam = vectorizer.vocabulary_

X_xval = vectorizer.transform(xval_dat['lines'])
X_test = vectorizer.transform(test_dat['lines'])
y_xval = [int(x) for x in xval_dat['labels']]
y_test = [int(x) for x in test_dat['labels']]

x2 = X_test

# fit naive bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

clf.score(X_train, y_train)
clf.score(X_xval, y_xval)
clf.score(X_test, y_test)
