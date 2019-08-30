from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame
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

# at_least_one = ["600", "610", "611", "630", "650", "500", "501", "502", "504", "505", "506", "507", "508",
#                         "510", "511",
#                         "513", "514", "515", "516", "518", "520", "521", "522", "524", "525", "526", "530", "532",
#                         "533", "534",
#                         "535", "536", "538", "540", "541", "542", "544", "545", "546", "547", "550", "552", "555",
#                         "556", "561",
#                         "562", "563", "565", "567", "580", "581", "583", "584", "585", "586", "588", "590", "595"]
at_least_one = ["505", "520", "521", "630", "650"]

class Classificator:
    def __init__(self, fields, vectorizer, undersample, model):
        date_now = datetime.now()
        self.results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        # self.pre = Preprocessor()
        # index = "records_mzk_filtered"

        # self.es = Elasticsearch()
        # s = Search(using=self.es, index=index)
        #
        # s = s.source(["001", "OAI", "072"] + at_least_one)
        #
        # s.execute()
        # dataframes = []
        # for hit in s.scan():
        #     hit_dict = hit.to_dict()
        #     if hit_dict.get("072", None) is None:
        #         continue
        #     if hit_dict.get("650", None) is None:
        #         continue
        #     if self.exists_at_least_one(hit_dict):
        #         try:
        #             df = self.transform_dict(hit_dict)
        #         except KeyError as error:
        #             continue
        #         dataframes.append(df)
        # self.data = pd.concat(dataframes)
        data_path = ""
        test_path = ""
        train_path = ""
        if fields == 'all':
            data_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\lem_all.csv"
            if undersample is True:
                train_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_all_under\\train.csv"
                test_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_all_under\\test.csv"
            else:
                train_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_all_all\\train.csv"
                test_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_all_all\\test.csv"
        elif fields == "select":
            self.data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\lem_select.csv",
                                    index_col=0)
            if undersample is True:
                train_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_select_under\\train.csv"
                test_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_select_under\\test.csv"
            else:
                train_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_select_all\\train.csv"
                test_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\bow_select_all\\test.csv"

        self.data = pd.read_csv(data_path, index_col=0)
        self.data = self.data.dropna(axis=0)
        self.test = pd.read_csv(test_path, index_col=0)
        self.train = pd.read_csv(train_path, index_col=0)

        self.vectorizer = Vectorizer(vectorizer=vectorizer, ngram=2)
        self.vectorizer.fit(self.data)
        # matrix = self.vectorizer.get_matrix(self.data)
        # self.data['vector'] = list(matrix)
        # self.vector = matrix

        self.train = self.data.loc[self.train['001'].tolist()]
        self.test = self.data.loc[self.test['001'].tolist()]

        self.model = model

    def to_matrix(self, text_matrix):
        rows = text_matrix.split('\n')
        if len(rows) < 50:
            print(len(rows))
        for row in rows:
            return row

    def exists_at_least_one(self, hit_dict):
        for key in at_least_one:
            if hit_dict.get(key, None) is not None:
                return True
        return False

    def transform_dict(self, hit_dict):
        new_dict = {}
        new_dict['001'] = hit_dict['001']
        new_dict['OAI'] = hit_dict['OAI']['a']
        if isinstance(hit_dict['072'], list):
            new_dict['konspekt'] = hit_dict['072'][0]['9']
        else:
            new_dict['konspekt'] = hit_dict['072']['9']
        text = ""
        for key in at_least_one:
            value = hit_dict.get(key, None)
            if value is None:
                continue
            else:
                if isinstance(value, list):
                    for a in value:
                        text = text + " " + str(a['a'])
                else:
                    text = text + " " + value['a']

        tokens = self.pre.remove_stop_words(text)
        tokens = self.pre.lemmatize(tokens)
        new_dict['text'] = ' '.join(tokens)
        df = DataFrame(new_dict, index=[hit_dict['001']])
        return df

    def save_state(self):
        self.save_dataframe(self.data, self.results_dir)
        self.vectorizer.save(self.results_dir)
        model_path = str(Path(self.results_dir) / "model.pickle")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)

    def save_model(self):
        try:
            os.makedirs(self.results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        model_path = str(Path(self.results_dir) / "model.pickle")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)

    def save_dataframe(self, dataframe, path=None):
        if path is None:
            date_now = datetime.now()
            results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        else:
            results_dir = path
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        dataframe.to_csv(results_dir + '/data.csv')

    def split_test_train(self, data):
        data = data[data.konspekt.isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                                            , '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'])]
        y = data.konspekt
        X = data.drop('konspekt', axis=1)
        return train_test_split(X, y, test_size=0.25, random_state=52, stratify=y)

    def undersample(self, X, y):
        rus = RandomUnderSampler()
        rus.fit_resample(X, y)
        indices = rus.sample_indices_
        return X.iloc[indices], y.iloc[indices]

    def save_test_train(self, x_train, x_test, y_train, y_test, path=None):
        x_test['konspekt'] = y_test
        x_train['konspekt'] = y_train
        if path is None:
            date_now = datetime.now()
            results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        else:
            results_dir = path
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        x_test.to_csv(results_dir + '/test.csv')
        x_train.to_csv(results_dir + '/train.csv')

    def fit(self):
        y = self.train['konspekt'].tolist()
        X = self.vectorizer.transform(self.train)
        print("start")
        self.model.fit(X, y)
        print("finish")

    def evaluate(self, save=False):
        y_real = self.test['konspekt'].tolist()
        X = self.vectorizer.transform(self.test)
        y_pred = self.model.predict(X)
        name = type(self.model).__name__
        params = self.model.get_params()
        score = precision_recall_fscore_support(y_real, y_pred, average='macro')
        print(name)
        print(params)
        print(score)
        if save is False:
            return
        try:
            os.makedirs(self.results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        result_path = str(Path(self.results_dir) / "result.txt")
        with open(result_path, 'w+') as file:
            file.write(str(name))
            file.write('\n')
            file.write(str(params))
            file.write('\n')
            file.write(str(score))
            file.write('\n')

# vectorizer = Vectorizer()
# tfidf = vectorizer.bag_of_words(data)
# print(list(tfidf.toarray()))
# te = TextExtractorPreTag('C:/Users/jakub/Documents/ziped1', 'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
# vectorizer = D2VVectorizer(data=te)
# vectorizer.save_model('./d2v.model')

#data['lematized'] = data['text'].apply(pre.lemmatize)
#data['vector'] = data['lematized'].apply(vectorizer.get_vector)
#data = data[~data.konspekt.isin(['6', '10', '13'])] odstranenie najmensich tried
clf = RandomForestClassifier()
# clf = LinearSVC(random_state=0, tol=1e-5)
# clf = MultinomialNB()
classificator = Classificator("all", "tfidf", True, clf)
classificator.fit()
classificator.evaluate(True)
# classificator.save_model()
# classificator.save_state()
# classificator.data = classificator.data.replace({'konspekt': '10'}, '6') # spojenie najmensich tried
# classificator.data = classificator.data.replace({'konspekt': '13'}, '6') # spojenie najmensich tried
# classificator.save_state()
# #classificator.save_dataframe(classificator.data)
# X_train, X_test, y_train, y_test = classificator.split_test_train(classificator.data)
# X_train, y_train = classificator.undersample(X_train, y_train)
# classificator.save_test_train(X_train, X_test, y_train, y_test)
