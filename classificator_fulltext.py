from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno
from preprocessor import Preprocessor
from helper.text_extractor import TextExtractorPre
from sklearn.model_selection import train_test_split, cross_val_score
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
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
from sklearn.svm import SVC

class Classificator:
    def __init__(self, index, model):
        date_now = datetime.now()
        self.results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))
        self.model = model
        self.index = index
        self.elastic = Elasticsearch(timeout=60)
        self. preprocessor = Preprocessor()
        self.counter = 0

    def generate_data(self, dictionary, process_documents=True, process_keywords=True):
        s = Search(using=self.elastic, index=self.index)
        s = s.params(scroll='5h', request_timeout=100)
        i = 0
        field = 'czech'
        documents = []
        keywords_rows = []
        dictionary = self.prepare_dictionary(dictionary)
        row = []
        col = []
        data = []
        shape = (0, 0)
        from_i = 35001
        to_i = 60000
        for hit in s.scan():
            if i < from_i:
                i += 1
                continue
            if i > to_i:
                break
            print("processing number: " + str(i))
            id_elastic = hit.meta.id
            print(id_elastic)
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
            non_keywords = self.non_keywords(term_vectors, hit['keywords'], 20)
            if process_documents:
                doc_row, tfidf_vector = self.document_row(hit, term_vectors, dictionary, doc_count, hit['czech_length'])
                documents.append(doc_row)
                row, col, data, shape = self.append_vector(row, col, data, shape, tfidf_vector)
            if process_keywords:
                kw_rows = self.keyword_rows(hit, term_vectors, non_keywords, doc_count, hit['czech_length'])
                keywords_rows.extend(kw_rows)
            i += 1

        if process_documents:
            data_docs = pd.concat(documents)
            Helper.save_dataframe(data_docs, 'documents', self.results_dir)
            sparse_matrix = csr_matrix((data, (row, col)), shape)
            Helper.save_sparse_matrix(sparse_matrix, self.results_dir, 'tfidf')
        if process_keywords:
            data_keywords = pd.concat(keywords_rows)
            Helper.save_dataframe(data_keywords, 'keywords', self.results_dir)

    def generate_data_keywords(self):
        s = Search(using=self.elastic, index=self.index)
        s = s.params(scroll='5h', request_timeout=100)
        i = 0
        field = 'czech'
        documents = []
        keywords_rows = []
        from_i = 0
        to_i = 999
        Helper.create_results_dir(self.results_dir)
        for hit in s.scan():
            if i < from_i:
                i += 1
                continue
            if i > to_i:
                break
            print("processing number: " + str(i))
            id_elastic = hit.meta.id
            print(id_elastic)
            hit = hit.to_dict()
            body = {
                "fields": [field],
                "positions": True,
                "term_statistics": True,
                "field_statistics": True,
                "filter": {
                    "max_num_terms": 3000,
                    "min_term_freq": 1,
                    "min_doc_freq": 1
                }
            }
            try:
                response = self.elastic.termvectors(self.index, id=id_elastic, body=body)
            except KeyError:
                continue
            doc_count = response['term_vectors'][field]['field_statistics']['doc_count']
            term_vectors = response['term_vectors'][field]['terms']
            # ,001,OAI,keywords,title
            doc_row = {'001': hit['id_mzk'], 'OAI': hit['oai'], 'keywords': ' '.join(hit['keywords']),
                       'title': hit['title']}
            documents.append(DataFrame(doc_row, index=[hit['id_mzk']]))
            kw_rows = self.keyword_rows_all(hit, term_vectors, doc_count, hit['czech_length'])
            data_keywords = pd.concat(kw_rows)
            if i == 0:
                data_keywords.to_csv(self.results_dir / 'keywords.csv', encoding='utf-8')
            else:
                with open(self.results_dir / 'keywords.csv', 'a', encoding='utf-8') as f:
                    data_keywords.to_csv(f, header=False, encoding='utf-8')
            i += 1

        data_docs = pd.concat(documents)
        Helper.save_dataframe(data_docs, 'documents', self.results_dir)

    def generate_dictionary(self):
        s = Search(using=self.elastic, index=self.index)
        s = s.params(scroll='5h', request_timeout=100)
        i = 0
        field = 'czech'
        from_i = 20000
        to_i = 25000
        words = []
        for hit in s.scan():
            if i < from_i:
                i += 1
                continue
            if i > to_i:
                break
            print("processing number: " + str(i))
            id_elastic = hit.meta.id
            print(id_elastic)
            body = {
                "fields": [field],
                "positions": True,
                "term_statistics": True,
                "field_statistics": True,
                "filter": {
                    "max_num_terms": 1,
                    "min_doc_freq": 100,
                    "max_doc_freq": 20000
                }
            }
            try:
                response = self.elastic.termvectors(self.index, id=id_elastic, body=body)
            except KeyError:
                continue
            term_vectors = response['term_vectors'][field]['terms']
            for word in term_vectors:
                if word not in words:
                    words.append(word)
                    # print(word)
            i += 1

        results_dir = Helper.create_results_dir()
        with open(results_dir / "slovnik.txt", "w+", encoding="utf-8") as file:
            for word in words:
                file.write(word)
                file.write('\n')


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
        row = {"id_mzk": id_mzk, "category": category, "group": group}
        return DataFrame(row, index=[id_mzk]), tfidf_vector

    def tfidf(self, term_freq, doc_freq,  doc_count, sum):
        inverse_doc_freq = np.log(doc_count/doc_freq)
        return term_freq / sum * inverse_doc_freq

    def keyword_rows(self, hit, term_vectors, non_keywords, doc_count, sum):
        id_mzk = hit['id_mzk']
        keywords = self.preprocessor.remove_stop_words(' '.join(hit['keywords']))
        keywords = self.preprocessor.lemmatize(keywords)
        new_set = set(keywords)
        keywords = list(new_set)

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
            df = DataFrame(new_word, index=[self.counter])
            self.counter += 1
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
            df = DataFrame(new_word, index=[self.counter])
            self.counter += 1
            result.append(df)

        return result

    def keyword_rows_all(self, hit, term_vectors, doc_count, sum):
        id_mzk = hit['id_mzk']

        title = self.preprocessor.remove_stop_words(hit['title'])
        title = self.preprocessor.lemmatize(title)

        result = []
        for word in term_vectors:
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

            new_word = {"id_mzk": id_mzk, "word": word, "tfidf": tfidf, "tag": tag,
                        "first_occurrence": first_occurrence/sum, "in_title": in_title}
            df = DataFrame(new_word, index=[self.counter])
            self.counter += 1
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

    def non_keywords(self, term_vectors, keywords, n_keys):

        keywords = self.preprocessor.remove_stop_words(' '.join(keywords))
        keywords = self.preprocessor.lemmatize(keywords)

        random_ints = []
        if n_keys > len(term_vectors):
            n_keys = len(term_vectors)
            for i in range(0, n_keys):
                random_ints.append(i)
        non_keywords = []

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

        return non_keywords

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

    def fit_eval(self, data, matrix, save=False):
        y = np.array(data['category'].tolist())
        X = matrix
        skf = StratifiedKFold(n_splits=2)
        skf.get_n_splits(X, y)
        i = 0
        precisions = []
        recalls = []
        fscores = []
        for train_index, test_index in skf.split(X, y):
            print("Start training iteration: " + str(i))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
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
            file.write(str(name) + '\n')
            file.write(str(params) + '\n')
            file.write(str(precisions) + '\n')
            file.write(str(sum(precisions)/len(precisions)) + '\n')
            file.write(str(recalls) + '\n')
            file.write(str(sum(recalls) / len(recalls)) + '\n')
            file.write(str(fscores) + '\n')
            file.write(str(sum(fscores) / len(fscores)) + '\n')

    def fit_eval2(self, data, matrix, save=False):
        y = np.array(data['category'].tolist())
        X = matrix
        skf = StratifiedKFold(n_splits=2)
        skf.get_n_splits(X, y)
        i = 0
        precisions = []
        recalls = []
        fscores = []
        train_index = []
        test_index = []
        with open("2019_10_25_16_18/indexes0.txt", 'r') as file:
            for line in file:
                if line[len(line) - 1] == '\n':
                    train_index.append(int(line[:-1]))
                else:
                    train_index.append(line)
        with open("2019_10_25_16_18/indexes1.txt", 'r') as file:
            for line in file:
                if line[len(line) - 1] == '\n':
                    test_index.append(int(line[:-1]))
                else:
                    test_index.append(line)
        print("Start training iteration: " + str(i))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
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
            file.write(str(name) + '\n')
            file.write(str(params) + '\n')
            file.write(str(precisions) + '\n')
            file.write(str(sum(precisions)/len(precisions)) + '\n')
            file.write(str(recalls) + '\n')
            file.write(str(sum(recalls) / len(recalls)) + '\n')
            file.write(str(fscores) + '\n')
            file.write(str(sum(fscores) / len(fscores)) + '\n')

    def grid_search(self, data, matrix):
        y = np.array(data['category'].tolist())
        X = matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # tuned_parameters = { 'alpha': np.linspace(0.5, 1.5, 6),
        #                      'fit_prior': [True, False]}
        # tuned_parameters = {'penalty': ['l1', 'l2'],
        #                     'loss': ['hinge', 'squared_hinge'],
        #                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
        #                     'tol': [1e-2, 1e-3, 1e-4, 1e-5]}

        # tuned_parameters = {
        #                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]}
        # tuned_parameters = {
        #     'bootstrap': [True],
        #     'max_depth': [80, 100],
        #     'max_features': [2, 3],
        #     'min_samples_leaf': [3, 5],
        #     'min_samples_split': [8, 12],
        #     'n_estimators': [300, 1000]
        # }
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10]}]
        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(self.model, tuned_parameters, cv=3,
                               scoring='%s_micro' % score, error_score=0.0, verbose=2, n_jobs=4)
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

# clf = LinearSVC(random_state=0, tol=1e-5, C=1)
# clf = MultinomialNB(alpha=0.5, fit_prior=True)
# clf = RandomForestClassifier(max_depth = 50, n_estimators=200, n_jobs=4)
# clf = SVC(kernel='rbf')
classificator = Classificator('fulltext_mzk', None)
# classificator.generate_dictionary()
# classificator.generate_data('keywords.txt')
classificator.generate_data_keywords()
# data = pd.read_csv('2019_10_22_09_54/documents.csv', index_col=0)
# matrix = load_npz('2019_10_22_09_54/tfidf.npz')
# classificator.fit_eval(data, matrix, save=True)
# classificator.grid_search(data, matrix)
# y = np.array(data['category'].tolist())
# X = matrix
# scores = cross_val_score(
#     clf, X, y, cv=10, scoring='f1_micro')
# print(scores)
# with open('uuids.txt', 'r') as file:
#     uuids = file.read()
#
# uuids = uuids.split('\n')
# parts = np.array_split(uuids, 4)
# te = TextExtractorPre('data/all', 'data/sorted_pages')
# for ui in parts[0]:
#     te.get_text(ui)

# results_dir = Helper.create_results_dir('2019_10_22_09_54')
# data1 = pd.read_csv('2019_10_21_15_06/documents.csv', index_col=0)
# data2 = pd.read_csv('2019_10_21_17_04/documents.csv', index_col=0)
# data3 = pd.read_csv('2019_10_21_18_38/documents.csv', index_col=0)
# data4 = pd.read_csv('2019_10_22_00_11/documents.csv', index_col=0)
# data = pd.concat([data1, data2, data3, data4])
# Helper.save_dataframe(data, 'documents', results_dir)

# matrix1 = load_npz('2019_10_22_09_54/tfidf.npz')
# print(matrix1.shape)
# matrix2 = load_npz('2019_10_21_17_04/tfidf.npz')
# matrix3 = load_npz('2019_10_21_18_38/tfidf.npz')
# matrix4 = load_npz('2019_10_22_00_11/tfidf.npz')
# matrix = vstack([matrix1, matrix2, matrix3, matrix4])
# Helper.save_sparse_matrix(matrix, results_dir, 'tfidf')