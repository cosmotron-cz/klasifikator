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


class KeywordsGeneratorTfidf:
    def fit_from_elastic(self, index):
        pairs = self.get_pairs()
        date_now = datetime.now()
        results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        es = Elasticsearch()
        s = Search(using=es, index=index)

        s.execute()
        dataframes = []
        for hit in s.scan():
            hit_dict = hit.to_dict()
            try:
                new_dict = {}
                new_dict['001'] = hit_dict['001']
                new_dict['OAI'] = hit_dict['OAI']['a']
                title = hit_dict.get('245', None)
                if title is not None:
                    new_dict['title'] = title.get('a', "") + title.get('b', "")
                df = DataFrame(new_dict, index=[hit_dict['001']])
            except KeyError:
                continue
            dataframes.append(df)
        data = pd.concat(dataframes)
        train, test = train_test_split(data, test_size=0.2)
        self.save_dataframe(train, 'train', results_dir)
        self.save_dataframe(test, 'test', results_dir)

        # preprocessor = Preprocessor()


        uuids = []
        for index, row in train.iterrows():
            # name_pre = ' '.join(preprocessor.lemmatize(row['title']))
            values = pairs.get(row['OAI'])
            uuids = uuids + values

        te = TextExtractorPre('C:/Users/jakub/Documents/ziped1',
                              'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages', uuids=uuids)


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
                        values.append(part)
                    else:
                        keys.append(part)
                for key in keys:
                    pairs[key] = values
        return pairs





index = "records_mzk_filtered" # TODO zmenit
kg = KeywordsGeneratorTfidf()
kg.fit_from_elastic(index)