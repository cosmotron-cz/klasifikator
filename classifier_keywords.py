from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno

from sklearn.preprocessing import LabelEncoder

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
from helper.helper import Helper


at_least_one = ["505", "520", "521", "630", "650"]


class ClassifierKeywords:
    def __init__(self, fields, vectorizer, undersample, model):
        date_now = datetime.now()
        self.results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        self.under = undersample
        self.fields = fields
        self.v = vectorizer
        self.pre = Preprocessor()
        index = "records_mzk_filtered"

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
        # Helper.save_dataframe(self.data, 'with_group_select', self.results_dir)
        data_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\2019_11_12_08_46\\with_group_select.csv"
        # if fields == 'all':
        #     data_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\lem_all.csv"
        # elif fields == "select":
        #     data_path = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\processed data\\lem_select.csv"
        self.data = pd.read_csv(data_path, index_col=0)
        self.data = self.data.dropna(axis=0)
        self.data = self.data[self.data.konspekt.isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                                                       '23', '24', '25', '26'])]

        self.vectorizer = Vectorizer(load_vec='2019_11_18_13_36/vectorizer.pickle')
        # self.vectorizer.fit(self.data)
        # self.vectorizer.save(self.results_dir)
        # labelencoder = LabelEncoder()
        # labelencoder.fit(self.data['group'])
        # Helper.save_model(labelencoder, self.results_dir, 'groups_labels')
        # Helper.save_model(self.vectorizer, self.results_dir, 'tfidf')
        # matrix = self.vectorizer.get_matrix(self.data)
        # self.data['vector'] = list(matrix)
        # self.vector = matrix

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
            new_dict['group'] = hit_dict['072'][0]['a']
        else:
            new_dict['konspekt'] = hit_dict['072']['9']
            new_dict['group'] = hit_dict['072']['a']
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
        if isinstance(X, (DataFrame, Series)):
            X_resample = X.iloc[indices]
        else:
            X_resample = X[indices]
        if isinstance(y, (DataFrame, Series)):
            y_resample = y.iloc[indices]
        else:
            y_resample = y[indices]
        return X_resample, y_resample

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
        score = precision_recall_fscore_support(y_real, y_pred, average='micro')
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

    def fit_eval(self, save=False):
        y = np.array(self.data['konspekt'].tolist())
        X = self.vectorizer.transform(self.data)
        model = LinearSVC(random_state=0, tol=1e-2, C=1)
        model.fit(X, y)
        Helper.save_model(model, self.results_dir, 'category')
        return
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, y)
        i = 0
        precisions = []
        recalls = []
        fscores = []
        for train_index, test_index in skf.split(X, y):
            print("Start training iteration: " + str(i))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if self.under:
                X_train, y_train = self.undersample(X_train, y_train)
            self.model.fit(X_train, y_train)
            print("End training")
            i += 1
            y_pred = self.model.predict(X_test)
            score = precision_recall_fscore_support(y_test, y_pred, average='micro')
            precisions.append(score[0])
            recalls.append(score[1])
            fscores.append(score[2])

        name = type(self.model).__name__
        params = self.model.get_params()
        print("fields:" + self.fields + " vectorizer: " + self.v + " undersample: " + str(self.under))
        print(name)
        print(params)
        print(precisions)
        print(str(sum(precisions)/len(precisions)))
        print(recalls)
        print(str(sum(recalls) / len(recalls)))
        print(fscores)
        print(str(sum(fscores) / len(fscores)))
        if save is False:
            return
        try:
            os.makedirs(self.results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        result_path = str(Path(self.results_dir) / "result.txt")
        with open(result_path, 'w+') as file:
            file.write("fields:" + self.fields + " vectorizer: " + self.v + " undersample: " + str(self.under) + '\n')
            file.write(str(name) + '\n')
            file.write(str(params) + '\n')
            file.write(str(precisions) + '\n')
            file.write(str(sum(precisions)/len(precisions)) + '\n')
            file.write(str(recalls) + '\n')
            file.write(str(sum(recalls) / len(recalls)) + '\n')
            file.write(str(fscores) + '\n')
            file.write(str(sum(fscores) / len(fscores)) + '\n')

    def fit_eval_groups(self, save=False):
        pd.options.mode.chained_assignment = None
        labelencoder = Helper.load_model('models/keywords/groups_labels.pickle')
        for i in range(1, 27):
            group = self.data[self.data['konspekt'] == i]
            # labelencoder = LabelEncoder()
            group['group'] = labelencoder.transform(group['group'])
            print(str(i) + " " + str(len(group.index)))
            y = np.array(group['group'].tolist())
            X = self.vectorizer.transform(group)
            model = LinearSVC(random_state=0, tol=1e-2, C=1)
            model.fit(X, y)
            Helper.save_model(model, self.results_dir, 'groups_' + str(i))
            continue
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(X, y)
            i = 0
            precisions = []
            recalls = []
            fscores = []
            for train_index, test_index in skf.split(X, y):
                print("Start training iteration: " + str(i))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if self.under:
                    X_train, y_train = self.undersample(X_train, y_train)
                self.model.fit(X_train, y_train)
                print("End training")
                i += 1
                y_pred = self.model.predict(X_test)
                score = precision_recall_fscore_support(y_test, y_pred, average='micro')
                precisions.append(score[0])
                recalls.append(score[1])
                fscores.append(score[2])

            name = type(self.model).__name__
            params = self.model.get_params()
            print("fields:" + self.fields + " vectorizer: " + self.v + " undersample: " + str(self.under))
            print(name)
            print(params)
            print(precisions)
            print(str(sum(precisions)/len(precisions)))
            print(recalls)
            print(str(sum(recalls) / len(recalls)))
            print(fscores)
            print(str(sum(fscores) / len(fscores)))
            if save is False:
                continue
            try:
                os.makedirs(self.results_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            name_file = "result" + str(i) + ".txt"
            result_path = str(Path(self.results_dir) / name_file)
            with open(result_path, 'w+') as file:
                file.write("fields:" + self.fields + " vectorizer: " + self.v + " undersample: " + str(self.under) + '\n')
                file.write(str(name) + '\n')
                file.write(str(params) + '\n')
                file.write(str(precisions) + '\n')
                file.write(str(sum(precisions)/len(precisions)) + '\n')
                file.write(str(recalls) + '\n')
                file.write(str(sum(recalls) / len(recalls)) + '\n')
                file.write(str(fscores) + '\n')
                file.write(str(sum(fscores) / len(fscores)) + '\n')

    def grid_search(self):
        y = np.array(self.data['konspekt'].tolist())
        X = self.vectorizer.transform(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # tuned_parameters = { 'alpha': np.linspace(0.5, 1.5, 6),
        #                      'fit_prior': [True, False]}
        # tuned_parameters = {'penalty': ['l1', 'l2'],
        #                     'loss': ['hinge', 'squared_hinge'],
        #                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
        #                     'tol': [1e-2, 1e-3, 1e-4, 1e-5]}

        # tuned_parameters = {
        #                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]}
        tuned_parameters = {'max_depth': [50],
                            'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}
        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(self.model, tuned_parameters, cv=4,
                               scoring='%s_micro' % score, error_score=0.0, verbose=2)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()


# vectorizer = Vectorizer()
# tfidf = vectorizer.bag_of_words(data)
# print(list(tfidf.toarray()))
# te = TextExtractorPreTag('C:/Users/jakub/Documents/ziped1', 'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
# vectorizer = D2VVectorizer(data=te)
# vectorizer.save_model('./d2v.model')

#data['lematized'] = data['text'].apply(pre.lemmatize)
#data['vector'] = data['lematized'].apply(vectorizer.get_vector)
#data = data[~data.konspekt.isin(['6', '10', '13'])] odstranenie najmensich tried
# clf = RandomForestClassifier()
clf = LinearSVC(random_state=0, tol=1e-2, C=1)
# clf = xgb.XGBClassifier()
# clf = MultinomialNB(alpha=0.5, fit_prior=True)
classificator = ClassifierKeywords("select", "tfidf", False, clf)
# classificator.grid_search()
classificator.fit_eval_groups()
# classificator.fit_eval(False)
# classificator.save_model()
# classificator.save_state()
# classificator.data = classificator.data.replace({'konspekt': '10'}, '6') # spojenie najmensich tried
# classificator.data = classificator.data.replace({'konspekt': '13'}, '6') # spojenie najmensich tried
# classificator.save_state()
# #classificator.save_dataframe(classificator.data)
# X_train, X_test, y_train, y_test = classificator.split_test_train(classificator.data)
# X_train, y_train = classificator.undersample(X_train, y_train)
# classificator.save_test_train(X_train, X_test, y_train, y_test)
