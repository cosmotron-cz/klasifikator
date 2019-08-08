from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


class Vectorizer():
    def tfidf(self, data, ngram=1, vocabulary=None):
        vectorizer = TfidfVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram))
        matrix = vectorizer.fit_transform(data['text'])
        return matrix

    def bag_of_words(self, data, ngram=1, vocabulary=None):
        vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram))
        matrix = vectorizer.fit_transform(data['text'])
        return matrix


class D2VVectorizer():
    def __init__(self, data=None, model=None):
        if data is None and model is None:
            raise Exception("No data or model for D2VVectorizer") # TODO po natrenovani dat default model
        if model is not None:
            if isinstance(model, str):
                self.model = Doc2Vec.load(model)
            else:
                self.model = model
            return
        if data is not None:
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
            self.model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
            self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def save_model(self, path):
        self.model.save(path)

    def get_vector(self, text):
        return self.model.infer_vector(text)
