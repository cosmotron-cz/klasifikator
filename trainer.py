from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno

from sklearn.preprocessing import LabelEncoder

from helper.config_handler import ConfigHandler
from preprocessor import Preprocessor
from vectorizer import Vectorizer
from elastic_handler import ElasticHandler
from data_import import DataImporter
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
from helper.helper import Helper
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
from sklearn.preprocessing import LabelEncoder, normalize


class Trainer:
    def __init__(self):
        models_dir = ConfigHandler.get_models_dir()
        date_now = datetime.now()
        self.results_dir = models_dir + "/" + date_now.strftime('%Y_%m_%d_%H_%M')
        try:
            os.makedirs(self.results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(self.results_dir + "/keywords")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(self.results_dir + "/fulltext")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.index = "training_" + date_now.strftime('%Y_%m_%d_%H_%M')
        print(self.index)

    def import_data(self, path):
        ElasticHandler.create_document_index(self.index)
        importer = DataImporter()
        importer.import_data(path, self.index)
        ElasticHandler.refresh(self.index)

    def train(self):
        trainer_keywords = TrainerKeywords(self.index, self.results_dir + "/keywords")
        trainer_keywords.generate_model()
        trainer_keywords.generate_model_groups()

        trainer_fulltext = TrainerFulltext(self.index, self.results_dir + "/fulltext")
        trainer_fulltext.generate_data()
        trainer_fulltext.generate_model()
        trainer_fulltext.generate_models_groups()

    def delete_index(self):
        ElasticHandler.remove_index(self.index)


class TrainerKeywords:
    def __init__(self, index, results_dir):
        self.results_dir = results_dir
        self.pre = Preprocessor()
        self.index = index

        self.es = Elasticsearch()
        self.vectorizer = Vectorizer(ngram=2)
        script_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.labelencoder = Helper.load_model(script_path / 'models/keywords/groups_labels.pickle')
        s = Search(using=self.es, index=index)

        s.execute()
        dataframes = []
        for hit in s.scan():
            hit_dict = hit.to_dict()
            konspekt = hit_dict.get("konspekt", None)
            if konspekt is None:
                print('None')
            if not konspekt:
                print('[]')
            if hit_dict.get("konspekt", None) is None or hit_dict.get("konspekt", None) == []:
                continue
            if hit_dict.get("keywords", None) is None or hit_dict.get("keywords", None) == []:
                continue
            df = self.transform_dict(hit_dict)
            dataframes.append(df)
        self.data = pd.concat(dataframes)
        self.data = self.data.dropna(axis=0)
        self.data = self.data[self.data.konspekt.isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                                                       '23', '24', '25', '26'])]

        self.vectorizer.fit(self.data)
        self.vectorizer.save(self.results_dir, "tfidf")

    def transform_dict(self, hit_dict):
        new_dict = {'001': hit_dict['id_001'], 'OAI': hit_dict['oai']}
        if isinstance(hit_dict['konspekt'], list):
            new_dict['konspekt'] = hit_dict['konspekt'][0]['category']
            new_dict['group'] = hit_dict['konspekt'][0]['group']
        else:
            new_dict['konspekt'] = hit_dict['konspekt']['category']
            new_dict['group'] = hit_dict['konspekt']['group']
        text = ' '.join(hit_dict['keywords'])
        additional_info = hit_dict.get('additional_info', "")
        if additional_info is not None:
            text += additional_info

        tokens = self.pre.remove_stop_words(text)
        tokens = self.pre.lemmatize(tokens)
        new_dict['text'] = ' '.join(tokens)
        df = DataFrame(new_dict, index=[hit_dict['id_001']])
        return df

    def generate_model(self):
        y = np.array(self.data['konspekt'].tolist())
        X = self.vectorizer.transform(self.data)
        model = LinearSVC(random_state=0, tol=1e-2, C=1)
        model.fit(X, y)
        Helper.save_model(model, self.results_dir, 'category')

    def generate_model_groups(self):
        pd.options.mode.chained_assignment = None

        for i in range(1, 27):
            group = self.data[self.data['konspekt'] == i]
            group['group'] = self.labelencoder.transform(group['group'])
            print(str(i) + " " + str(len(group.index)))
            y = np.array(group['group'].tolist())
            X = self.vectorizer.transform(group)
            model = LinearSVC(random_state=0, tol=1e-2, C=1)
            model.fit(X, y)
            Helper.save_model(model, self.results_dir, 'groups_' + str(i))


class TrainerFulltext:
    def __init__(self, index, result_dir):

        self.results_dir = result_dir
        self.index = index
        self.elastic = Elasticsearch(timeout=60)
        self.preprocessor = Preprocessor()
        self.counter = 0
        self.data = None
        self.matrix = None
        self.script_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.labelencoder = Helper.load_model(self.script_path / 'models/keywords/groups_labels.pickle')

    def generate_data(self):
        # TODO filtorvat len tie zaznamy ktore maju 072
        s = Search(using=self.elastic, index=self.index)
        s = s.params(scroll='5h', request_timeout=100)
        i = 0
        field = 'text'
        documents = []
        dictionary = self.prepare_dictionary(str(self.script_path / 'dictionary.txt'))
        row = []
        col = []
        data = []
        shape = (0, 0)
        for hit in s.scan():
            print("processing number: " + str(i))
            id_elastic = hit.meta.id
            print(id_elastic)
            hit = hit.to_dict()
            term_vectors, doc_count = ElasticHandler.term_vectors(self.index, id_elastic)
            if term_vectors is not None and doc_count is not None:
                doc_row, tfidf_vector = self.document_row(hit, term_vectors, dictionary, doc_count, hit['text_length'])
                documents.append(doc_row)
                row, col, data, shape = self.append_vector(row, col, data, shape, tfidf_vector)
            i += 1

        self.data = pd.concat(documents)
        self.matrix = csr_matrix((data, (row, col)), shape)

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
        new_set = set(res)
        res = list(new_set)
        res.sort()
        return res

    def document_row(self, hit, term_vectors, dictionary, doc_count, sum):
        id_mzk = hit['id_001']
        konspekt = hit['konspekt']
        if isinstance(konspekt, list):
            category = konspekt[0]['category']
            group = konspekt[0]['group']
        else:
            category = konspekt['category']
            group = konspekt['group']

        tfidf_vector = []
        for word in dictionary:
            term = term_vectors.get(word, None)
            if term is None:
                tfidf_vector.append(0.0)
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, sum)
                tfidf_vector.append(tfidf)
        row = {"id_mzk": id_mzk, "category": category, "group": group}
        return DataFrame(row, index=[id_mzk]), tfidf_vector

    def tfidf(self, term_freq, doc_freq,  doc_count, sum):
        inverse_doc_freq = np.log(doc_count/doc_freq)
        return term_freq / sum * inverse_doc_freq

    def append_vector(self, row, col, data, shape, tfidf_vector):
        if shape[1] == 0:
            shape = (shape[0], len(tfidf_vector))
        for i, tfidf in enumerate(tfidf_vector):
            if tfidf > 0.0:
                data.append(tfidf)
                row.append(shape[0])
                col.append(i)
        shape = (shape[0] + 1, shape[1])
        return row, col, data, shape

    def generate_model(self):
        self.data.reset_index(drop=True, inplace=True)
        y = np.array(self.data['category'].tolist())
        X = normalize(self.matrix)
        model = LinearSVC(random_state=0, tol=1e-2, C=1)
        model.fit(X, y)
        Helper.save_model(model, self.results_dir, 'category')

    def generate_models_groups(self):
        self.data.reset_index(drop=True, inplace=True)
        for i in range(1, 27):
            group = self.data[self.data['category'] == i]
            group['category'] = self.labelencoder.transform(group['group'])
            print(str(i) + " " + str(len(group.index)))
            group_matrix = self.matrix[group.index, :]
            y = np.array(group['category'].tolist())
            X = normalize(group_matrix)
            model = LinearSVC(random_state=0, tol=1e-2, C=1)
            model.fit(X, y)
            Helper.save_model(model, self.results_dir, 'groups_' + str(i))
