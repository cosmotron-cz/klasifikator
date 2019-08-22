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

# at_least_one = ["600", "610", "611", "630", "650", "500", "501", "502", "504", "505", "506", "507", "508",
#                         "510", "511",
#                         "513", "514", "515", "516", "518", "520", "521", "522", "524", "525", "526", "530", "532",
#                         "533", "534",
#                         "535", "536", "538", "540", "541", "542", "544", "545", "546", "547", "550", "552", "555",
#                         "556", "561",
#                         "562", "563", "565", "567", "580", "581", "583", "584", "585", "586", "588", "590", "595"]
at_least_one = ["505", "520", "521", "630", "650"]

class Classificator:
    def __init__(self):
        date_now = datetime.now()
        self.results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        self.pre = Preprocessor()
        index = "records_mzk_filtered"

        self.es = Elasticsearch()
        s = Search(using=self.es, index=index)

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
        self.data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\lemmatize_selected.csv", index_col=0)
        self.data = self.data.dropna(axis=0)
        self.vectorizer = Vectorizer(vectorizer="bow", ngram=2)
        matrix = self.vectorizer.get_matrix(self.data)
        self.data['vector'] = list(matrix)
        self.model = "test"

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
        vec_path = str(Path(self.results_dir) / "model.pickle")
        with open(vec_path, "wb") as file:
            pickle.dump(self.vectorizer, file)

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

        dataframe.to_csv(results_dir + '/medzivysledok.csv')

    def split_test_train(self, data):
        y = data.konspekt
        X = data.drop('konspekt', axis=1)
        return train_test_split(X, y, test_size=0.33, random_state=52)

    def undersample(self, X, y):
        rus = RandomUnderSampler()
        rus.fit_resample(X, y)
        indices = rus.sample_indices_
        return X.iloc[indices], y.iloc[indices]



# vectorizer = Vectorizer()
# tfidf = vectorizer.bag_of_words(data)
# print(list(tfidf.toarray()))
te = TextExtractorPreTag('C:/Users/jakub/Documents/ziped1', 'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
vectorizer = D2VVectorizer(data=te)
vectorizer.save_model('./d2v.model')

#data['lematized'] = data['text'].apply(pre.lemmatize)
#data['vector'] = data['lematized'].apply(vectorizer.get_vector)
#data = data[~data.konspekt.isin(['6', '10', '13'])] odstranenie najmensich tried
# classificator = Classificator()
# classificator.save_state()
# classificator.data = classificator.data.replace({'konspekt': '10'}, '6') # spojenie najmensich tried
# classificator.data = classificator.data.replace({'konspekt': '13'}, '6') # spojenie najmensich tried
# classificator.save_state()
# #classificator.save_dataframe(classificator.data)
# X_train, X_test, y_train, y_test = classificator.split_test_train(classificator.data)
# X_resampled, y_resampled = classificator.undersample(X_train, y_train)
# print(X_resampled)
#print(data)
#print(new_data)
