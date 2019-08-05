from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


class Vectorizer():
    def tfidf(self, data, ngram=1, vocabulary=None):
        vectorizer = TfidfVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram))
        matrix = vectorizer.fit_transform(data['text'])
        return matrix

    def bag_of_words(self, data, ngram=1, vocabulary=None):
        vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram))
        matrix = vectorizer.fit_transform(data['text'])
        return matrix