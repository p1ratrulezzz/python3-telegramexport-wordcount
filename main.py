# loading in all the essentials for data manipulation
import pandas as pd
import numpy as np
# load inthe NTLK stopwords to remove articles, preposition and other words that are not actionable
from nltk.corpus import stopwords
# This allows to create individual objects from a bog of words
from nltk.tokenize import word_tokenize
# Lemmatizer helps to reduce words to the base form
from nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
from nltk import ngrams
# We can use counter to count the objects
from collections import Counter
# This is our visual library
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pathlib
import matplotlib.figure

def word_frequency(sentence):
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    new_tokens = word_tokenize(sentence)
    new_tokens = [t.lower() for t in new_tokens]
    # new_tokens = [t for t in new_tokens if t not in stopwords.words('russian')]
    new_tokens = [t for t in new_tokens if (t.isalpha() and len(t) >= 4)]
    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
    # counts the words, pairs and trigrams
    counted = Counter(new_tokens)
    counted_2 = Counter(ngrams(new_tokens, 2))
    counted_3 = Counter(ngrams(new_tokens, 3))
    # creates 3 data frames and returns thems
    word_freq = pd.DataFrame(counted.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)
    word_pairs = pd.DataFrame(counted_2.items(), columns=['pairs', 'frequency']).sort_values(by='frequency',
                                                                                             ascending=False)
    trigrams = pd.DataFrame(counted_3.items(), columns=['trigrams', 'frequency']).sort_values(by='frequency',
                                                                                              ascending=False)
    return word_freq, word_pairs, trigrams

cwd = os.path.dirname(__file__)
fp = open(cwd  + '/resources/result.json')
data = json.load(fp)
fp.close()

sentences = ""
for message in data["messages"]:
    if (type(message["text"]) == str and len(message["text"]) > 0):
        sentences += message["text"] + " "

(data2, data3, data4) = word_frequency(sentences)
a = 1
# # create subplot of the different data frames
fig: matplotlib.figure.Figure
fig, axes = plt.subplots(2,1,figsize=(20,40))
sns.barplot(ax=axes[0],x='frequency',y='word',data=data2.head(100))
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data3.head(50))
#sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data4.head(30))

fig.savefig(cwd + '/resources/fig.jpg')