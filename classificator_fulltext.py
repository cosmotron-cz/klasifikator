from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno
from preprocessor import Preprocessor
from helper.text_extractor import TextExtractorPre
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
import random
from helper.helper import Helper
from multiprocessing import Pool
from scipy.sparse import csr_matrix

class Classificator:
    def __init__(self, index, model):
        date_now = datetime.now()
        self.results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))
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
            id_elastic = hit.meta.id
            hit = hit.to_dict()
            body = {
                "fields": [field],
                "positions": True,
                "term_statistics": True,
                "field_statistics": True
            }
            try:
                response = self.elastic.termvectors(self.index, id=id_elastic, body=body)
            except KeyError:
                continue
            doc_count = response['term_vectors'][field]['field_statistics']['doc_count']
            term_vectors = response['term_vectors'][field]['terms']
            non_keywords, sum = self.non_keywords_and_terms_sum(term_vectors, hit['keywords'], 20)
            if process_documents:
                doc_row = self.document_row(hit, term_vectors, dictionary, doc_count, sum)
                documents.append(doc_row)
            if process_keywords:
                kw_rows = self.keyword_rows(hit, term_vectors, non_keywords, doc_count, sum)
                keywords_rows.extend(kw_rows)

        data_docs = pd.concat(documents)
        Helper.save_dataframe(data_docs, 'documents', self.results_dir)
        data_keywords = pd.concat(keywords_rows)
        Helper.save_dataframe(data_keywords, 'keywords', self.results_dir)

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

        res = self.preprocessor.tokenize(' '.join(dictionary))
        res = self.preprocessor.lemmatize(res)
        return res

    def document_row(self, hit, term_vectors, dictionary, doc_count, sum):
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
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, sum)
                tfidf_vector.append(tfidf)
        row = {"id_mzk": id_mzk, "category": category, "group": group, "tfidf": str(tfidf_vector)}
        return DataFrame(row)

    def tfidf(self, term_freq, doc_freq,  doc_count, sum):
        inverse_doc_freq = np.log(doc_count/doc_freq)
        return term_freq / sum * inverse_doc_freq

    def keyword_rows(self, hit, term_vectors, non_keywords, doc_count, sum):
        id_mzk = hit['id_mzk']
        keywords = self.preprocessor.remove_stop_words(' '.join(hit['keywords']))
        keywords = self.preprocessor.lemmatize(keywords)

        title = self.preprocessor.remove_stop_words(hit['title'])
        title = self.preprocessor.lemmatize(title)

        result = []
        for word in keywords:
            term = term_vectors.get(word, None)
            if term is None:
                continue
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, sum)
                first_occurrence = term['tokens'][0]['position']
            tag = self.preprocessor.pos_tag(word)[0][0]
            if word in title:
                in_title = True
            else:
                in_title = False

            new_word = {"id_mzk": id_mzk, "word": word, "is_keyword": 1, "tfidf": tfidf, "tag": tag,
                        "first_occurrence": first_occurrence/sum, "in_title": in_title}
            df = DataFrame(new_word, index=[1])
            result.append(df)

        for word in non_keywords:
            term = term_vectors.get(word, None)
            if term is None:
                continue
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, sum)
                first_occurrence = term['tokens'][0]['position']
            tag = self.preprocessor.pos_tag(word)[0][0]
            if word in title:
                in_title = True
            else:
                in_title = False

            new_word = {"id_mzk": id_mzk, "word": word, "is_keyword": 0, "tfidf": tfidf, "tag": tag,
                        "first_occurrence": first_occurrence/sum, "in_title": in_title}
            df = DataFrame(new_word, index=[1])
            result.append(df)

        return result

    def non_keywords_and_terms_sum(self, term_vectors, keywords, n_keys):

        keywords = self.preprocessor.remove_stop_words(' '.join(keywords))
        keywords = self.preprocessor.lemmatize(keywords)

        non_keywords = []
        sum = 0
        random_ints = []
        while len(random_ints) != n_keys:
            random_i = random.randint(0, len(term_vectors))
            if random_i not in random_ints:
                random_ints.append(random_i)

        for i, key in enumerate(term_vectors):
            if i in random_ints:
                if key in keywords:
                    random_ints.append(random.randint(i+1, len(term_vectors)))
                else:
                    non_keywords.append(key)

            sum += term_vectors[key]['term_freq']

        return non_keywords, sum


# classificator = Classificator('test_text', None)
# classificator.generate_data('keywords.txt')


with open('uuids.txt', 'r') as file:
    uuids = file.read()

uuids = uuids.split('\n')
parts = np.array_split(uuids, 4)
te = TextExtractorPre('data/all', 'data/sorted_pages')
for ui in parts[3]:
    te.get_text(ui)
