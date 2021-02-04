import errno
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import pickle
from pathlib import Path
from pandas import DataFrame, Series

class Vectorizer():
    def __init__(self, vectorizer='tfidf', ngram=1, vocabulary=None, load_vec=None, input='content'):
        if load_vec is not None:
            with open(load_vec, "rb") as file:
                self.vectorizer = pickle.load(file)
            return
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram), input=input,
                                              token_pattern=r"(?u)\S\S+")
        elif vectorizer == 'bow':
            self.vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, ngram), input=input)
        else:
            raise Exception("Unknown vectorizer")

    def fit(self, data):
        if isinstance(data, DataFrame):
            self.vectorizer.fit(data['text'])
        else:
            self.vectorizer.fit(data)

    def transform(self, data):
        if isinstance(data, DataFrame):
            matrix = self.vectorizer.transform(data['text'])
        else:
            matrix = self.vectorizer.transform(data)
        return matrix

    def get_matrix(self, data):
        if isinstance(data, DataFrame):
            matrix = self.vectorizer.fit_transform(data['text'])
        else:
            matrix = self.vectorizer.fit_transform(data)
        return matrix

    def save(self, path, name=None):
        if name is not None:
            if not name.endswith('.pickle'):
                name += ".pickle"
            vec_path = str(Path(path) / name)
        else:
            vec_path = str(Path(path) / "vectorizer.pickle")
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        with open(vec_path, "wb") as file:
            pickle.dump(self.vectorizer, file)


class D2VVectorizer():
    def __init__(self, data=None, model=None):
        assert gensim.models.doc2vec.FAST_VERSION > -1
        if data is None and model is None:
            raise Exception("No data or model for D2VVectorizer") # TODO po natrenovani dat default model
        if model is not None:
            if isinstance(model, str):
                self.model = Doc2Vec.load(model)
            else:
                self.model = model
            return
        if data is not None:
            print("start training")
            self.model = Doc2Vec(data, vector_size=300, dm=1, window=3, min_count=1, epochs=10, workers=8)
            # print("building vocabulary")
            # self.model.build_vocab(data)
            # print("star training")
            # self.model.train(data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            # self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def save_model(self, path):
        self.model.save(path)
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def get_vector(self, text):
        return self.model.infer_vector(text)
