import nltk
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


# remaining stopwords after cutting off last 50~ words of "from nltk.corpus import stopwords" since they had sentiment value (don't, should've, etc)
stopwords = ['not', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
             'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
             'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
             'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
             'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
             'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
             'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
             'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just']

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

with open('./ratings/unlabeled.txt', 'r') as f:
    unlabeled_rating_list = [line.rstrip('\n').split()[-1] for line in f]

unlabeled_rating_df = pd.DataFrame(unlabeled_rating_list, columns=['rating'])

neg_list = []
for entry in os.scandir(neg_dir):
    with open(entry, 'r', encoding='latin-1') as f:  # encoding param here is a hack, TODO: ask Celia
        goodWords = []

        # lemmatize string
        cleanString = lemmatize_sentence(f.read())

        # remove irrelevant words from review
        wordList = f.read().split()
        for word in wordList:
            if word.lower() not in stopwords:
                goodWords.append(word)
        cleanString = ' '.join(goodWords)

        cur = [cleanString, 'neg']
        neg_list.append(cur)

pos_list = []
for entry in os.scandir(pos_dir):
    with open(entry, 'r', encoding='latin-1') as f:  # encoding param here is a hack, TODO: ask Celia
        goodWords = []

        # lemmatize string
        cleanString2 = lemmatize_sentence(f.read())

        # remove irrelevant words from review
        wordList = cleanString2.split()
        for word in wordList:
            if word.lower() not in stopwords:
                goodWords.append(word)
        cleanString2 = ' '.join(goodWords)

        cur = [cleanString2, 'pos']
        pos_list.append(cur)

unlabeled_list = []
for entry in os.scandir(pos_dir):
    with open(entry, 'r', encoding='latin-1') as f:  # encoding param here is a hack, TODO: ask Celia
        unlabeledWords = []

        # lemmatize string
        cleanString2 = lemmatize_sentence(f.read())

        # remove irrelevant words from review
        wordList = cleanString2.split()
        for word in wordList:
            if word.lower() not in stopwords:
                unlabeledWords.append(word)
        cleanString2 = ' '.join(unlabeledWords)

        cur = [cleanString2]
        unlabeled_list.append(cur)

# combining dataframes
neg_df = pd.DataFrame(neg_list, columns=['review', 'label'])
neg_df = neg_df.join(neg_rating_df)
pos_df = pd.DataFrame(pos_list, columns=['review', 'label'])
pos_df = pos_df.join(pos_rating_df)

frames = [neg_df, pos_df]
full_df = pd.concat(frames)

unlabeled_df = pd.DataFrame(unlabeled_list, columns=['review'])
unlabeled_df = unlabeled_df.join(unlabeled_rating_df)

# Okay cool! Now that our data is in a much easier shape, we can start building the model
model = LogisticRegression()
vectorizer = CountVectorizer()
X = full_df['review']
y = full_df['label']

features = vectorizer.fit_transform(X)

# split data into 5 folds
scores = cross_val_score(model, features, y, cv=5)
print(np.mean(scores))

model.fit(features, y)
rating_features = vectorizer.transform(unlabeled_df['review'])
rating_pred = model.predict(rating_features)

# write flagged, fake reviews into separate .txt file
with open("flagged_reviews.txt", "a") as r:
    for i in range(0, len(rating_pred)):
        if rating_pred[i] == 'pos' and (
                unlabeled_df['rating'].iloc[i] == '1.0' or unlabeled_df['rating'].iloc[i] == '2.0'):
            r.write(unlabeled_df['review'].iloc[i] + '-- {}'.format(unlabeled_df['rating'].iloc[i]) + '\n' + '\n')
            # if test_pred[i] == 'neg' and (z_test.iloc[i] == '4.0' or z_test.iloc[i] == '5.0'):
            #     r.write(X_test.iloc[i] + '-- {}'.format(z_test.iloc[i]) + '\n' + '\n')

# print('Accuracy score: ', accuracy_score(y_test, test_pred))
# print('Precision score: ', precision_score(y_test, test_pred, pos_label='pos'))
# print('Recall score: ', recall_score(y_test, test_pred, pos_label='pos'))
# print('F-1 score: ', f1_score(y_test, test_pred, pos_label='pos'))
# print(classification_report(y_test, test_pred))
