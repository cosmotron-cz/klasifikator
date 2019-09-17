from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
import errno
from preprocessor import Preprocessor
from helper.text_extractor import TextExtractorPre
from vectorizer import Vectorizer

class KeywordsGeneratorTfidf:
    def fit_from_elastic(self, index):
        pairs = self.get_pairs()
        all_files = []
        directory = 'C:/Users/jakub/Documents/ziped1'
        for file in os.listdir(directory):
            if file.endswith(".tar.gz"):
                file = file[:-7]
                all_files.append(file)
        date_now = datetime.now()
        results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        es = Elasticsearch()
        s = Search(using=es, index=index)

        s.execute()
        dataframes = []
        need = 100
        have = 0
        for hit in s.scan():
            hit_dict = hit.to_dict()
            try:
                new_dict = {}
                new_dict['001'] = hit_dict['001']
                new_dict['OAI'] = hit_dict['OAI']['a']
                # TODO pridat 650 na porovnanie
                values = pairs.get(new_dict['OAI'])
                if values is not None:
                    found = False
                    for uuid in values:
                        if uuid in all_files:
                            found = True
                            break
                    if found is False:
                        continue
                else:
                    continue
                title = hit_dict.get('245', None)
                if title is not None:
                    new_dict['title'] = title.get('a', "") + title.get('b', "")
                df = DataFrame(new_dict, index=[hit_dict['001']])
                have += 1
                if need == have:
                    break
            except KeyError:
                continue
            dataframes.append(df)
        data = pd.concat(dataframes)
        train, test = train_test_split(data, test_size=0.2)
        self.save_dataframe(train, 'train', results_dir)
        self.save_dataframe(test, 'test', results_dir)

        uuids = []
        for index, row in train.iterrows():
            # name_pre = ' '.join(preprocessor.lemmatize(row['title']))
            values = pairs.get(row['OAI'])
            if values is not None:
                uuids = uuids + values

        te = TextExtractorPre(directory,
                              'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages', uuids=uuids)
        texts = []
        for text in te:
            texts.append(text)

        vectorizer = Vectorizer(vectorizer='tfidf', ngram=1)
        vectorizer.fit(te)
        vectorizer.save(results_dir)

    def extract_keywords(self, tfidf_vectorizer, text):
        feature_names = tfidf_vectorizer.vectorizer.get_feature_names()
        vector = tfidf_vectorizer.transform(text)
        inverse_vector = vectorizer.vectorizer.inverse_transform(vector)
        print(inverse_vector)
        sorted_items = self.sort_coo(vector.tocoo())
        keywords = self.extract_topn_from_vector(feature_names, sorted_items, 10)
        for k in keywords:
            print(k, keywords[k])

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        # use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        # results = zip(feature_vals,score_vals)
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results


    def save_dataframe(self, dataframe, name, path=None):
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

        dataframe.to_csv(results_dir + '/' + name + '.csv')

    def get_pairs(self):
        path = 'C:/Users/jakub/Documents/sloucena_id'

        pairs = {}
        with open(path, 'r') as file:
            for line in file:
                line = line.rstrip()
                parts = line.split(',')
                keys = []
                values = []
                for part in parts:
                    if part.startswith('uuid'):
                        values.append(part.replace(':', '_'))
                    else:
                        keys.append(part)
                for key in keys:
                    pairs[key] = values
        return pairs





index = "records_mzk_filtered" # TODO zmenit
kg = KeywordsGeneratorTfidf()
# kg.fit_from_elastic(index)
with open("C:\\Users\\jakub\\Documents\\ziped1\\processed\\uuid_1ef7b740-f6e3-11e8-a5a4-005056827e52.txt",
          'r', encoding="utf-8") as file:
    text = [file.read()]

vectorizer = Vectorizer(load_vec='2019_09_17_14_33\\vectorizer.pickle')
kg.extract_keywords(vectorizer, text)
