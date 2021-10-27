# loading in all the essentials for data manipulation
import pandas as pd
from nltk import ngrams
# We can use counter to count the objects
from collections import Counter
# This is our visual library
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import matplotlib.figure
import spacy
from spacy.tokens.token import Token
from progress.bar import Bar
from threading import Thread
from time import sleep
import sys

class ProgressBarThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        super(ProgressBarThread, self).__init__(group=None, target=None, name=None,
                                                args=(), kwargs=kwargs, daemon=None)
        self._ctx = args[0]
        self._progress = Bar('Processing', max=self._ctx["max"])
    def run(self) -> None:
        self._progress.start()
        previous_i = self._ctx["i"]
        while (self._ctx["i"] <= self._ctx["max"]):
            delta = self._ctx["i"] - previous_i
            previous_i = self._ctx["i"]
            if delta > 0:
                self._progress.next(delta)
            sleep(1)

        self._progress.finish()

def word_frequency(messages):
    ctx = {
        "i": 0,
        "max": len(messages)
    }
    progressthread = ProgressBarThread(args=[ctx])
    progressthread.start()
    sp = spacy.load('ru_core_news_lg')
    new_tokens = []
    for message in messages:
        if (type(message["text"]) == str and len(message["text"]) > 0):
            sentence = sp(message["text"])
            word: spacy.tokens.token.Token
            for word in sentence:
                if (word.is_alpha and len(word.lemma_) > 3):
                    new_tokens.append(word.lemma_)
        ctx["i"] += 1
    progressthread.join()

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

cwd = os.path.dirname(os.path.realpath(__file__))
fp = open(cwd  + '/resources/result.json')
data = json.load(fp)
fp.close()

(data2, data3, data4) = word_frequency(data["messages"])
# # create subplot of the different data frames
fig: matplotlib.figure.Figure
fig, axes = plt.subplots(3,1,figsize=(40,40))
sns.barplot(ax=axes[0],x='frequency',y='word',data=data2.head(100))
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data3.head(50))
sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data4.head(30))

fig.savefig(cwd + '/resources/fig.jpg')