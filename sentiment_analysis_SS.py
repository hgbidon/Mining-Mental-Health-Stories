import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys

# Mental Health Stories from Portraits of Resilence
mental_health_story_1 = open('sathyasilva.txt', 'r').read()
sentences = sent_tokenize(mental_health_story_1)

sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
