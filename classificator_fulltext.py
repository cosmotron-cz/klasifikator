from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno
from preprocessor import Preprocessor
from vectorizer import Vectorizer, D2VVectorizer
from helper.text_extractor import TextExtractorPreTag
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pickle
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb

class Classificator:
    def __init__(self, index, model):
        date_now = datetime.now()
        self.results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        self.model = model
        self.index = index
        self.elastic = Elasticsearch()
        self. preprocessor = Preprocessor()

    def generate_data(self, dictionary, process_documents=True, process_keywords=True):
        s = Search(using=self.elastic, index=self.index)
        s.execute()
        i = 0
        field = 'czech'
        documents = []
        keywords_rows = []
        dictionary = self.prepare_dictionary(dictionary)
        for hit in s.scan():
            hit = hit.to_dict()
            body = {
                "fields": [field],
                "positions": True,
                "term_statistics": True,
                "field_statistics": True
            }
            response = self.elastic.termvectors(self.index, id='XA44ym0BMO8lDpHzO3Y8', body=body)
            doc_count = response['term_vectors'][field]['field_statistics']['doc_count']
            term_vectors = response['term_vectors']['terms']
            if process_documents:
                doc_row = self.document_row(hit, term_vectors, dictionary, doc_count)
                documents.append(doc_row)
            if process_keywords:
                kw_rows = self.keyword_rows(hit, term_vectors, doc_count)
                keywords_rows.extend(kw_rows)
            # TODO pridat ukladanie dataframu a tfidf matice pre doc_rows

    def prepare_dictionary(self, dictionary):
        if isinstance(dictionary, str):
            dict = []
            with open(dictionary, 'r', encoding='utf-8') as f:
                for line in f:
                    if line[len(line) - 1] == '\n':
                        dict.append(line[:-1])
                    else:
                        dict.append(line)
            dictionary = dict
        elif not isinstance(dictionary, list):
            raise Exception('Wrong type for dict: ' + str(dictionary))

        res = self.elastic.indices.analyze(index=self.index,
                                           body={"analyzer": "czech", "text": ' '.join(dictionary)})
        tokens = []
        for token in res['tokens']:
            if token['token'] not in tokens:
                tokens.append(token['token'])
        return tokens

    def document_row(self, hit, term_vectors, dictionary, doc_count):
        # TODO zjednotit dictionary - ak sa bude spustat viac krat,
        #  v jednom behu to nie je problem kedze dictionary bude rovnaky
        id_mzk = hit['id_mzk']
        konpsket = hit['konpsket']
        if isinstance(konpsket, list):
            category = konpsket[0]['category']
            group = konpsket[0]['group']
        else:
            category = konpsket['category']
            group = konpsket['group']

        tfidf_vector = []
        for word in dictionary:
            term = term_vectors.get(word, None)
            if term is None:
                tfidf_vector.append(0.0)
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count)
                tfidf_vector.append(tfidf)
        row = {"id_mzk": id_mzk, "category": category, "group": group, "tfidf": tfidf_vector}
        return row

    def tfidf(self, term_freq, doc_freq,  doc_count):
        # TODO skusit zistik ako efektivne vypocitat dlzku dokumentu
        inverse_doc_freq = np.log(doc_count/doc_freq)
        return term_freq * inverse_doc_freq

    def keyword_rows(self, hit, term_vectors, doc_count):
        id_mzk = hit['id_mzk']
        res = self.elastic.indices.analyze(index=self.index,
                                           body={"analyzer": "czech", "text": ' '.join(hit['keywords'])})
        keyowrds = []
        for word in res['tokens']:
            if word['token'] not in keyowrds:
                keyowrds.append(word['token'])

        res = self.elastic.indices.analyze(index=self.index,
                                           body={"analyzer": "czech", "text": ' '.join(hit['title'])})
        title = []
        for word in res['tokens']:
            if word['token'] not in title:
                title.append(word['token'])

        for word in keyowrds:
            term = term_vectors.get(word, None)
            if term is None:
                continue
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count)
                first_occurrence = term['tokens'][0]['position']
            tag = self.preprocessor.pos_tag(word)[0] # TODO skontrolovat
            if word in title:
                in_title = True
            else:
                in_title = False

            new_word = {"id_mzk": id_mzk, "word": word, "is_keyword": True, "tfidf": tfidf, "tag": tag,
                        "first_occurrence": first_occurrence, "in_title": in_title}


        # TODO pridat ne klucove slova
