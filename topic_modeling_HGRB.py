import nltk
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
from nltk.corpus import stopwords
import spacy

import matplotlib.pyplot as plt
import seaborn as sns

mental_health_story_1 = open('hanagabriellebidon.txt', 'r').read()
sentences = sent_tokenize(mental_health_story_1)
sentences_dict = {}
for index in range(len(sentences)):
    element = sentences[index]
    key = 'sentence_' + str(index)
    sentences_dict[key] = element
sentences_df = pd.DataFrame.from_dict(sentences_dict, orient='index')

# Get the 20 most frequent words in the mental health story
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

# remove unwanted characters, numbers and symbols
sentences_df[0] = sentences_df[0].str.replace("[^a-zA-Z#]", " ")
stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(rev):
  sentences_new = " ".join([i for i in rev if i not in stop_words])
  return sentences_new
# remove short words (length < 3)
sentences_df[0] = sentences_df[0].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# remove stopwords from the text
story = [remove_stopwords(r.split()) for r in sentences_df[0]]

nlp = spacy.load('en_core_web_sm')
# Get the 20 most frequent words in the mental health story
def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_story = pd.Series(story).apply(lambda x: x.split())
print(tokenized_story)
print(len(tokenized_story))
story_2 = lemmatization(tokenized_story)
print(story_2)
print(len(story_2))
important_words_dict = {}
story_3 = []
for i in range(len(story_2)):
    story_3.append(' '.join(story_2[i]))

sentences_df[0] = story_3
print(sentences_df[0])

frequency_of_words = freq_words(sentences_df[0], 35)
