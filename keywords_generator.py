from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import os
import errno
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocessor import Preprocessor
from helper.text_extractor import TextExtractorPre, TextExtractor
from rouge import Rouge
from multi_rake import Rake
from pathlib import Path
from lxml import etree
from helper.helper import Helper
from preprocessor import Preprocessor
from vectorizer import Vectorizer
from imblearn.under_sampling import RandomUnderSampler
import pickle
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class KeywordsGeneratorTfidf:
    def fit_from_elastic(self, index):
        pairs = self.get_pairs()
        all_files = []
        directory = 'data/all'
        for file in os.listdir(directory):
            if file.endswith(".tar.gz"):
                file = file[:-7]
                all_files.append(file)
        date_now = datetime.now()
        results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        # es = Elasticsearch()
        # s = Search(using=es, index=index)
        #
        # s.execute()
        # dataframes = []
        # need = 10000
        # have = 0
        # for hit in s.scan():
        #     hit_dict = hit.to_dict()
        #     try:
        #         new_dict = {}
        #         new_dict['001'] = hit_dict['001']
        #         new_dict['OAI'] = hit_dict['OAI']['a']
        #         field_650 = hit_dict['650']
        #         keywords = ""
        #         if isinstance(field_650, list):
        #             for field in field_650:
        #                 keywords += " " + field.get('a', "")
        #         else:
        #             keywords = field_650.get('a', "")
        #         new_dict['keywords'] = keywords
        #         values = pairs.get(new_dict['OAI'])
        #         if values is not None:
        #             found = False
        #             for uuid in values:
        #                 if uuid in all_files:
        #                     found = True
        #                     break
        #             if found is False:
        #                 continue
        #         else:
        #             continue
        #         title = hit_dict.get('245', None)
        #         if title is not None:
        #             new_dict['title'] = title.get('a', "") + title.get('b', "")
        #         df = DataFrame(new_dict, index=[hit_dict['001']])
        #         have += 1
        #         if need == have:
        #             break
        #     except KeyError:
        #         continue
        #     dataframes.append(df)
        # data = pd.concat(dataframes)
        # train, test = train_test_split(data, test_size=0.2)
        # self.save_dataframe(train, 'train', results_dir)
        # self.save_dataframe(test, 'test', results_dir)

        train = pd.read_csv('train2.csv', index_col=0)
        train = train.iloc[:1000]
        # test = pd.read_csv('test.csv', index_col=0)
        uuids = []
        for indx, row in train.iterrows():
            # name_pre = ' '.join(preprocessor.lemmatize(row['title']))
            values = pairs.get(row['OAI'])
            if values is not None:
                uuids = uuids + values

        te = TextExtractorPre(directory,
                              'data/sorted_pages', uuids=uuids, filter_nouns=False)

        vocabulary = self.preprocess_kw()
        vectorizer = Vectorizer(vectorizer='tfidf', ngram=4, vocabulary=vocabulary)
        vectorizer.fit(te)
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        vectorizer.save(results_dir)

    def tfidf_keywords(self, data, vectorizer):
        date_now = datetime.now()
        results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))

        pairs = self.get_pairs()
        preprocessor = Preprocessor()
        directory = Path('data/all')
        extractor = TextExtractor(directory,
                                  'data/sorted_pages')

        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        keywords = []
        i = 0
        for index, row in data.iterrows():
            print("processing number: " + str(i))
            values = pairs.get(row['OAI'], None)
            if values is not None:
                uuid = values[0]
                print(uuid)
                text = self.check_processed(uuid, directory)
                if text is None:
                    text = extractor.get_text(uuid)
                    text = preprocessor.remove_stop_words(text)
                    text = preprocessor.lemmatize(text)
                    text = ' '.join(text)
                    self.save_document(text, uuid, directory)
                # text = text.split(' ')
                # text = Helper.filter_words(text, preprocessor)
                # text = ' '.join(text)
                print("end preprocessing")
                kw = self.extract_keywords(vectorizer, text)
                kw2 = []
                for word in kw:
                    kw2.extend(word.split('_'))
                result = ' '.join(kw2)
                keywords.append(result)
            else:
                keywords.append("")
            i += 1

        data['generated'] = keywords

        self.save_dataframe(data, 'test', results_dir)

    def tfidf_keywords2(self, data, vectorizer):
        date_now = datetime.now()
        results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))

        # preprocessor = Preprocessor()

        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        keywords = []
        for index, row in data.iterrows():
            kw = self.extract_keywords(vectorizer, row['proccesed_content'])
            result = ' '.join(kw)
            keywords.append(result)

        data['generated'] = keywords

        self.save_dataframe(data, 'test', results_dir)

    def extract_keywords(self, tfidf_vectorizer, text):
        feature_names = tfidf_vectorizer.vectorizer.get_feature_names()
        vector = tfidf_vectorizer.transform([text])
        sorted_items = self.sort_coo(vector.tocoo())
        keywords = self.extract_topn_from_vector(feature_names, sorted_items, 6)
        return keywords

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):

        sorted_items = sorted_items[:topn]
        results = []

        for idx, score in sorted_items:
            results.append(feature_names[idx])

        return results

    def save_dataframe(self, dataframe, name, path=None):
        if path is None:
            date_now = datetime.now()
            results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))
        elif isinstance(path, Path):
            results_dir = path
        elif isinstance(path, str):
            results_dir = Path(path)
        else:
            raise Exception("wrong path argument")
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        if not name.endswith('.csv'):
            name += '.csv'
        dataframe.to_csv(results_dir / name, encoding='utf-8')

    def get_pairs(self):
        return Helper.get_pairs('data/sloucena_id')

    def rake_keywords(self, data, cont=False):
        date_now = datetime.now()
        results_dir = Path("rake")
        rake = Rake(language_code='cs')
        preprocessor = Preprocessor()
        keywords = []
        if cont:
            for indx, row in data.iterrows():
                text = preprocessor.lemmatize(row['content'])
                text = ' '.join(text)
                kw = rake.apply(text)
                result = ""
                for word in kw[:10]:
                    result += " " + word[0]
                # print(result)
                keywords.append(result)
        else:
            pairs = self.get_pairs()
            extractor = TextExtractor('C:/Users/jakub/Documents/all',
                                      'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')

            try:
                os.makedirs(results_dir / "processed")
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            i = 0
            for index, row in data.iterrows():
                print("processing number: " + str(i))
                values = pairs.get(row['OAI'], None)
                if values is not None:
                    uuid = values[0]
                    print(uuid)
                    text = self.check_processed(uuid, results_dir)
                    if text is None:
                        print("preprocessing")
                        text = extractor.get_text(uuid)
                        text = preprocessor.lemmatize(text)
                        text = ' '.join(text)
                        self.save_document(text, uuid, results_dir)
                    print("end preprocessing")
                    kw = rake.apply(text)
                    result = ""
                    for word in kw[:10]:
                        result += " " + word[0]
                    # print(result)
                    keywords.append(result)
                else:
                    keywords.append("")
                i += 1

        data['generated'] = keywords

        self.save_dataframe(data, 'test_contents', results_dir)

    def count_keywords_in_text(self, data):
        date_now = datetime.now()
        results_dir = Path("rake")
        preprocessor = Preprocessor()
        pairs = self.get_pairs()
        extractor = TextExtractor('C:/Users/jakub/Documents/all',
                                  'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')

        try:
            os.makedirs(results_dir / "processed")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        i = 0
        all_n = 0
        all_keywords = 0
        for index, row in data.iterrows():
            # print("processing number: " + str(i))
            values = pairs.get(row['OAI'], None)
            if values is not None:
                uuid = values[0]
                # print(uuid)
                text = self.check_processed(uuid, results_dir)
                if text is None:
                    # print("preprocessing")
                    text = extractor.get_text(uuid)
                    text = preprocessor.lemmatize(text)
                    text = ' '.join(text)
                    self.save_document(text, uuid, results_dir)
                # print("end preprocessing")
                keywords = row['keywords']
                keywords = preprocessor.lemmatize(keywords)
                text = text.split(' ')
                n = 0
                len_keywords = len(keywords)
                for word in text:
                    if len(keywords) == 0:
                        break
                    found = None
                    for indx, key in enumerate(keywords):
                        if word == key:
                            n += 1
                            found = indx
                            break
                    if found is not None:
                        keywords.pop(found)
                print("for document " + str(i) + " found " + str(n) + " out of " + str(len_keywords))
                all_n += n
                all_keywords += len_keywords
            i += 1
        print("for all documents found " + str(all_n) + " out of " + str(all_keywords))
        print()

    def save_document(self, document, current_uuid, result_dir):
        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        dir = Path(result_dir)
        file = dir / 'processed' / current_uuid
        with open(file, 'w+', encoding="utf-8") as f:
            f.write(document)

    def check_processed(self, current_uuid, directory):

        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        file = directory / 'processed' / current_uuid
        if file.is_file():
            with open(file, 'r', encoding="utf-8") as f:
                document = f.read()
        else:
            return None
        return document

    def evaluate_keywords(self, dataframe):
        preprocessor = Preprocessor()
        rouge = Rouge(metrics=["rouge-1", "rouge-2"])
        recall_1 = 0.0
        precision_1 = 0.0
        f1_1 = 0.0
        recall_2 = 0.0
        precision_2 = 0.0
        f1_2 = 0.0
        number = 0
        for indx, row in dataframe.iterrows():
            number += 1
            keywords = row['keywords']
            keywords = preprocessor.remove_stop_words(keywords)
            keywords = preprocessor.lemmatize(keywords)
            keywords = ' '.join(keywords)
            try:
                scores = rouge.get_scores(keywords, row['generated'])
            except AttributeError as ae:
                scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
            recall_1 += scores[0]['rouge-1']['r']
            precision_1 += scores[0]['rouge-1']['p']
            f1_1 += scores[0]['rouge-1']['f']
            recall_2 += scores[0]['rouge-2']['r']
            precision_2 += scores[0]['rouge-2']['p']
            f1_2 += scores[0]['rouge-2']['f']

        print("rouge 1 average scores")
        print("recall: " + str(recall_1 / number))
        print("preccision: " + str(precision_1 / number))
        print("f1: " + str(f1_1 / number))

        print("rouge 2 average scores")
        print("recall: " + str(recall_2 / number))
        print("preccision: " + str(precision_2 / number))
        print("f1: " + str(f1_2 / number))

    def only_small_files(self):
        path = Path("C:\\Users\\jakub\Documents\\all\\processed")
        pairs = self.get_pairs()
        train = pd.read_csv('train.csv', index_col=0)
        indexes = []
        i = 0
        for indx, row in train.iterrows():
            add = False
            values = pairs.get(row['OAI'])
            if values is not None:
                uuid = values[0]
                if not uuid.endswith('.txt'):
                    uuid += '.txt'
                file = path / uuid
                if file.is_file():
                    size = os.path.getsize(file)
                    if 200000 < size < 1000000:
                        add = True
            if add:
                indexes.append(i)
            i += 1
        train = train.iloc[indexes, :]
        print(train)
        self.save_dataframe(train, 'train2', '.')

    def read_contents(self):
        toc = Path('C:\\Users\\jakub\Documents\\toc.xml')
        # parser = etree.XMLParser(strip_cdata=False)
        tree = etree.parse(str(toc))
        contents = {}
        for book in tree.getroot():
            isbn = None
            content = None
            for child in book:
                if child.tag == 'bibinfo':
                    for childchild in child:
                        if childchild.tag == 'isbn':
                            isbn = childchild.text
                if child.tag == 'toc':
                    content = child.text
            if isbn is not None and content is not None:
                contents[isbn] = content

        return contents

    def add_contents(self, data, index):
        es = Elasticsearch()
        isbns = []
        contents = []
        for indx, row in data.iterrows():
            q = Q({"match": {"001": row['001']}})
            s = Search(using=es, index=index).query(q)
            s.execute()
            for hit in s:
                hit_dict = hit.to_dict()
                field_020 = hit_dict.get('020', None)
                if field_020 is None:
                    isbns.append('')
                    break
                if isinstance(field_020, list) and field_020[0].get('a', None) is not None and \
                        isinstance(field_020[0]['a'], str):
                    isbns.append(field_020[0]['a'])
                elif not isinstance(field_020, list) and field_020.get('a', None) is not None and \
                        isinstance(field_020['a'], str):
                    isbns.append(field_020['a'])
                else:
                    isbns.append('')
                break
        data['isbn'] = isbns
        indexes = []
        obalky_contents = self.read_contents()
        i = 0
        for indx, row in data.iterrows():
            content = obalky_contents.get(row['isbn'], '')
            contents.append(content)
            if content != '':
                indexes.append(i)
            i += 1

        data['content'] = contents
        data = data.iloc[indexes, :]
        self.save_dataframe(data, 'train_contents')

    def create_keywords_dict(self, index):
        es = Elasticsearch()
        s = Search(using=es, index=index)

        s.execute()
        keywords = {}
        for hit in s.scan():
            hit_dict = hit.to_dict()
            try:
                field_650 = hit_dict['650']
                if isinstance(field_650, list):
                    for field in field_650:
                        if field.get('a', "") == "":
                            continue
                        if field.get('2', '') != 'czenas':
                            continue
                        keywords[field['a']] = True
                else:
                    if field_650.get('2', '') != 'czenas':
                        continue
                    if field_650.get('a', "") != "":
                        keywords[field_650['a']] = True
            except KeyError:
                continue

        result = []
        for key in keywords:
            result.append(key)
        result.sort()
        for key in result:
            print(key)

    def read_keywords(self):
        file = 'keywords.txt'

        keywords = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line[len(line) - 1] == '\n':
                    keywords.append(line[:-1])
                else:
                    keywords.append(line)
        return keywords

    def preprocess_kw(self, keywords=None):
        if keywords is None:
            keywords = self.read_keywords()
        preprocessor = Preprocessor()
        result = []
        for key in keywords:
            parenthesis = key.find('(')
            if parenthesis != -1:
                parts = key.split('(')
                key = parts[0]
                if key[len(key) - 1] == ' ':
                    key = key[:-1]
            key = key.split(' ')
            if len(key) > 4:
                continue
            key = preprocessor.lemmatize(key)
            result.append(' '.join(key))
        result = list(dict.fromkeys(result))
        return result

    def elastic_keywords(self, index, dictionary=None):
        if dictionary is not None:
            dictionary = self.prepare_dictionary(dictionary)
        preprocessor = Preprocessor()
        elastic = Elasticsearch()
        s = Search(using=elastic, index=index)
        s = s.params(scroll='5h')
        i = 0
        field = 'czech'
        results = []
        for hit in s.scan():
            print("processing number: " + str(i))
            id_elastic = hit.meta.id
            hit = hit.to_dict()
            body = {
                "fields": ["czech"],
                "term_statistics": True,
                "field_statistics": True,
                "positions": False,
                "offsets": False,
                "filter": {
                    "max_num_terms": 500,
                    "min_term_freq": 1,
                    "min_doc_freq": 1
                }
            }
            try:
                response = elastic.termvectors(index, id=id_elastic, body=body)
            except KeyError:
                continue
            term_vectors = response['term_vectors'][field]['terms']
            keywords = preprocessor.remove_stop_words(' '.join(hit['keywords']))
            keywords = preprocessor.lemmatize(keywords)
            new_set = set(keywords)
            keywords = list(new_set)
            keywords = ' '.join(keywords)
            generated = []
            for word in term_vectors:
                tag = preprocessor.pos_tag(word)[0][0]
                if tag == 'A' or tag == 'N':
                    if dictionary is not None:
                        if word in dictionary:
                            generated.append(word)
                    else:
                        generated.append(word)
                    if len(generated) == 10:
                        break
            generated = ' '.join(generated)
            new_dict = {"001": hit['id_mzk'], 'OAI': hit['oai'], 'keywords': keywords, 'title': hit['title'],
                        'generated': generated}
            results.append(DataFrame(new_dict, index=[hit['id_mzk']]))
            i += 1

        data = pd.concat(results)
        results_dir = Helper.create_results_dir()
        Helper.save_dataframe(data, 'keywords_elastic_tfidf', results_dir)

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

    def fit_eval(self, data, under, model, save=False):
        y = np.array(data['is_keyword'].tolist())
        X = data[['tfidf', 'first_occurrence']]
        temp = data[['tag', 'in_title']]
        temp = temp.apply(LabelEncoder().fit_transform)
        X['tag'] = temp['tag']
        X['in_title'] = temp['in_title']
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        skf.get_n_splits(X, y)
        i = 0
        precisions = []
        recalls = []
        fscores = []
        for train_index, test_index in skf.split(X, y):
            print("Start training iteration: " + str(i))
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if under:
                X_train, y_train = self.undersample(X_train, y_train)
            model.fit(X_train, y_train)
            print("End training")
            i += 1
            y_pred = model.predict(X_test)
            score = precision_recall_fscore_support(y_test, y_pred, average='micro')
            precisions.append(score[0])
            recalls.append(score[1])
            fscores.append(score[2])

        name = type(model).__name__
        params = model.get_params()
        print("undersample: " + str(under))
        print(name)
        print(params)
        print(precisions)
        print(str(sum(precisions) / len(precisions)))
        print(recalls)
        print(str(sum(recalls) / len(recalls)))
        print(fscores)
        print(str(sum(fscores) / len(fscores)))
        if save is False:
            return
        results_dir = Helper.create_results_dir()
        result_path = str(Path(results_dir) / "result.txt")
        with open(result_path, 'w+') as file:
            file.write("undersample: " + str(under) + '\n')
            file.write(str(name) + '\n')
            file.write(str(params) + '\n')
            file.write(str(precisions) + '\n')
            file.write(str(sum(precisions) / len(precisions)) + '\n')
            file.write(str(recalls) + '\n')
            file.write(str(sum(recalls) / len(recalls)) + '\n')
            file.write(str(fscores) + '\n')
            file.write(str(sum(fscores) / len(fscores)) + '\n')

    def fit_save_model(self, data, under, model, exclude=None):
        if exclude is not None:
            exclude_ids = exclude['001'].tolist()
            data = data[~data.id_mzk.isin(exclude_ids)]
        y = np.array(data['is_keyword'].tolist())
        X = data[['tfidf', 'first_occurrence']]
        temp = data[['tag', 'in_title']]
        results_dir = Helper.create_results_dir()
        labelencoder = LabelEncoder()
        list1 = temp['tag'].tolist()
        list2 = temp['in_title'].tolist()
        list1.extend(list2)
        labelencoder.fit(list1)
        Helper.save_model(labelencoder, results_dir, 'labels')
        temp = temp.apply(labelencoder.transform)
        X['tag'] = temp['tag']
        X['in_title'] = temp['in_title']
        if under:
            X, y = self.undersample(X, y)
        print("Start training")
        model.fit(X, y)
        print("End training")
        Helper.save_model(model, results_dir, 'keywords_minus_1000')

    def generate_keywords_from_model(self, documents, keywords, model, labelencoder):
        keywords['tag'] = labelencoder.transform(keywords['tag'])
        keywords['in_title'] = labelencoder.transform(keywords['in_title'])
        document_keywords = []
        for index_i, row in documents.iterrows():
            doc_words = keywords[keywords['id_mzk'] == row['001']]
            if len(doc_words.index) == 0:
                document_keywords.append('')
                continue
            X = doc_words[['tfidf', 'first_occurrence', 'tag', 'in_title']]
            y_pred = model.predict(X)
            kw = []
            j = 0
            for index_j, word_row in doc_words.iterrows():
                if y_pred[j] == 1:
                    kw.append(word_row['word'])
                j += 1
            document_keywords.append(' '.join(kw))
        results_dir = Helper.create_results_dir()
        documents['generated'] = document_keywords
        Helper.save_dataframe(documents, 'keywords_1000_randomforest', results_dir)

    def generate_keywords_from_model_reduced(self, documents, keywords, model, labelencoder, kw_len=10):
        keywords['tag'] = labelencoder.transform(keywords['tag'])
        keywords['in_title'] = labelencoder.transform(keywords['in_title'])
        document_keywords = []
        for index_i, row in documents.iterrows():
            doc_words = keywords[keywords['id_mzk'] == row['001']]
            if len(doc_words.index) == 0:
                document_keywords.append('')
                continue
            X = doc_words[['tfidf', 'first_occurrence', 'tag', 'in_title']]
            y_pred = model.predict(X)
            kw = []
            j = 0
            for index_j, word_row in doc_words.iterrows():
                if y_pred[j] == 1:
                    kw.append(word_row)
                j += 1
            kw = pd.concat(kw, axis=1).T
            kw['tag'] = labelencoder.inverse_transform(kw['tag'].astype(int))
            kw['in_title'] = labelencoder.inverse_transform(kw['in_title'].astype(int))
            if len(kw.index) > kw_len:
                kw = kw[kw.tag.isin(['A', 'N'])]
            if len(kw.index) > kw_len:
                # kw.sort_values(by=['tfidf'], inplace=True, ascending=False)
                # kw = kw.head(kw_len)
                kw_in_title = kw[kw['in_title'] == 'True']
                kw_not_in_title = kw[kw['in_title'] == 'False']

                if len(kw_in_title.index) < kw_len:
                    kw_not_in_title.sort_values(by=['first_occurrence'], inplace=True)
                    kw_not_in_title = kw_not_in_title.head(kw_len - len(kw_in_title.index))
                    kw = pd.concat([kw_in_title, kw_not_in_title])
                else:
                    kw_in_title.sort_values(by=['first_occurrence'], inplace=True)
                    kw = kw_in_title.head(kw_len)

            document_keywords.append(' '.join(kw['word'].tolist()))
        results_dir = Helper.create_results_dir()
        documents['generated'] = document_keywords
        Helper.save_dataframe(documents, 'keywords_1000_randomforest', results_dir)

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

        preprocessor = Preprocessor()
        res = preprocessor.tokenize(' '.join(dictionary))
        res = preprocessor.lemmatize(res)
        new_set = set(res)
        res = list(new_set)
        res.sort()
        return res

    def count_found_keywords(self, index):
        elastic = Elasticsearch(timeout=60)
        s = Search(using=elastic, index=index)
        s = s.params(scroll='5h', request_timeout=100)
        preprocessor = Preprocessor()
        i = 0
        field = 'czech'
        keywords_count = 0
        found_count = 0
        from_i = 0
        to_i = 1000
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
                    "max_num_terms": 4000,
                    "min_term_freq": 1,
                    "min_doc_freq": 1
                }
            }
            try:
                response = elastic.termvectors(index, id=id_elastic, body=body)
            except KeyError:
                continue
            term_vectors = response['term_vectors'][field]['terms']
            keywords = preprocessor.remove_stop_words(' '.join(hit['keywords']))
            keywords = preprocessor.lemmatize(keywords)
            keywords_count += len(keywords)
            selected_words = []
            for word in term_vectors:
                selected_words.append(word)
            for word in keywords:
                if word in selected_words:
                    found_count += 1
            i += 1
        print(keywords_count)
        print(found_count)
        print(found_count/keywords_count)


index = "fulltext_mzk"
kg = KeywordsGeneratorTfidf()
# kg.elastic_keywords(index, 'keywords.txt')
# kg.count_found_keywords(index)
# documents = pd.read_csv('2019_10_30_10_35/documents.csv', index_col=0)
# preprocessor = Preprocessor()
# keywords_count = 0
# found_count = 0
# for index_i, row in documents.iterrows():
#     original = preprocessor.remove_stop_words(row['keywords'])
#     original = preprocessor.lemmatize(original)
#     keywords_count += len(original)
#     generated = row['generated'].split(' ')
#     for word in original:
#         if word in generated:
#             found_count += 1
#
# print(keywords_count)
# print(found_count)
# print(found_count/keywords_count)
# keywords = pd.read_csv('2019_10_30_10_35/keywords.csv', index_col=0)
# keywords = keywords[keywords['is_keyword'] == 0]
# plt.figure()
# keywords['tfidf'].plot.hist(bins=10)
# sns.catplot(x="in_title", kind="count", data=keywords)
# plt.show()
# model = Helper.load_model('2019_10_28_15_28/keywords_minus_1000.pickle')
# labels = Helper.load_model('2019_10_28_15_28/labels.pickle')
# kg.generate_keywords_from_model_reduced(documents, keywords, model, labels)
# kg.elastic_keywords(index=index)
# kg.create_keywords_dict(index)
# kg.read_keywords()
# kg.preprocess_kw()
# test = pd.read_csv('2019_10_30_16_22/keywords_1000_randomforest.csv', index_col=0)
# test = test.iloc[:1000]
# rake_test = pd.read_csv('rake\\test.csv', index_col=0)
# test['rake_keywords'] = rake_test['generated']
# kg.save_dataframe(test, 'results', None)
# test = test.iloc[:1000]
# kg.count_keywords_in_text(test)
# kg.rake_keywords(test, False)


# kg.fit_from_elastic(index)
test = pd.read_csv('2019_10_31_17_29/keywords_elastic_tfidf.csv', index_col=0)
kg.evaluate_keywords(test)
# clf = LinearSVC(random_state=0, tol=1e-5, C=1)
# clf = SVC(kernel='poly')
# clf = RandomForestClassifier(max_depth=50, n_estimators=200, n_jobs=4)
# exclude = pd.read_csv('2019_10_25_16_57/documents.csv', index_col=0)
# kg.fit_eval(test, True, clf)
# kg.fit_save_model(test, True, clf, exclude)
# vectorizer = Vectorizer(load_vec='2019_10_08_14_34/tfidf_vocabkw_1000.pickle')
# kg.tfidf_keywords(test, vectorizer)

# kg.add_contents(test, index)
# kg.read_contents()
# contents = test['content']
# preprocessor = Preprocessor()
# processed_contents = []
# for content in contents:
#     content = preprocessor.remove_stop_words(content)
#     content = preprocessor.lemmatize(content)
#     content = ' '.join(content)
#     processed_contents.append(content)
# test['proccesed_content'] = processed_contents
# date_now = datetime.now()
# results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
# # kg.save_dataframe(test, 'proccesed_contents')
# vectorizer = Vectorizer(vectorizer='tfidf', ngram=2)
# vectorizer.fit(test['proccesed_content'])
# try:
#     os.makedirs(results_dir)
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
# vectorizer.save(results_dir)
