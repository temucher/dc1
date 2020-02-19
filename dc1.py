import nltk
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Need to iterate through neg and pos folders, pulling each review into a dataframe
# 1 dataframe per label, including rating

neg_dir = './neg'
pos_dir = './pos'
unlabeled_dir = './unlabeled'

# get all ratings into a list to make them easier to match up to their corresponding review
with open('./ratings/negative.txt', 'r') as f:
    # have to do some weird thing here because the format of ratings data
    neg_rating_list = [line.rstrip('\n').split()[-1] for line in f]

# put this into rating dataframe, to be combined later
neg_rating_df = pd.DataFrame(neg_rating_list, columns=['rating'])

with open('./ratings/positive.txt', 'r') as f:
    pos_rating_list = [line.rstrip('\n').split()[-1] for line in f]

pos_rating_df = pd.DataFrame(pos_rating_list, columns=['rating'])


neg_list = []
for entry in os.scandir(neg_dir):
    with open(entry, 'r', encoding='latin-1') as f: # encoding param here is a hack, TODO: ask Celia
        cur = [f.read(), 'neg']
        neg_list.append(cur)

pos_list = []
for entry in os.scandir(pos_dir):
    with open(entry, 'r', encoding='latin-1') as f: # encoding param here is a hack, TODO: ask Celia
        cur = [f.read(), 'pos']
        pos_list.append(cur)

# combining dataframes
neg_df = pd.DataFrame(neg_list, columns=['review', 'label'])
neg_df = neg_df.join(neg_rating_df)

pos_df = pd.DataFrame(pos_list, columns=['review', 'label'])
pos_df = pos_df.join(pos_rating_df)

frames = [neg_df, pos_df]
full_df = pd.concat(frames)

# print(full_df.head())

# Okay cool! Now that our data is in a much easier shape, we can start building the model
model = LogisticRegression()
# split data into train, test sets
X = full_df['review']
# print(X)
y = full_df['label']
X_train, X_test, y_train, y_test= train_test_split(X, y)

# Tokenize review column
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(X_train)
test_features = vectorizer.transform(X_test)
# print(train_features)
# print(vectorizer.vocabulary_)

model.fit(train_features, y_train)
test_pred = model.predict(test_features)
# print(test_pred)

print('Accuracy score: ', accuracy_score(y_test, test_pred))
print('Precision score: ', precision_score(y_test, test_pred, pos_label='pos'))
print('Recall score: ', recall_score(y_test, test_pred, pos_label='pos'))
print('F-1 score: ', f1_score(y_test, test_pred, pos_label='pos'))
print(classification_report(y_test, test_pred))
