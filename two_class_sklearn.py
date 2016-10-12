# two_class_sklearn.py

"""
implement a basic text classifier to assign if the speaker of a given sentence was Donald Trump or Hillary Clinton

based on: 
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py
http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
"""

import pickle
import itertools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# load processed lines
with open('clinton.pickle', 'rb') as f : clinton = pickle.load(f)
with open('trump.pickle', 'rb') as f : trump = pickle.load(f)

# set up data frame to store lines and labels
# note labels are: 0 (clinton), 1 (trump) 
all_clinton = list(itertools.chain.from_iterable(clinton))
all_trump = list(itertools.chain.from_iterable(trump))
all_lines = all_clinton + all_trump 
labels = np.repeat(np.arange(2), [len(all_clinton), len(all_trump)])
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
# vectorizer.get_feature_names()
# vectorizer.vocabulary_

# create feature and label matrices/vectors for cross validation and test sets
X_xval = vectorizer.transform(xval_dat['lines'])
X_test = vectorizer.transform(test_dat['lines'])
y_xval = [int(x) for x in xval_dat['labels']]
y_test = [int(x) for x in test_dat['labels']]

# fit naive bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

clf.score(X_train, y_train) # seet training accuracy
clf.score(X_xval, y_xval) # see cross validation accuracy (tweak model based on this)
clf.score(X_test, y_test) # once model is FIXED, test on this
