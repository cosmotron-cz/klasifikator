from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
import errno
from preprocessor import Preprocessor
from helper.text_extractor import TextExtractorPre, TextExtractor
from vectorizer import Vectorizer
from rouge import Rouge
from pathlib import Path
from lxml import etree
from io import StringIO
from helper.helper import Helper

class CrfKeywords:
    def generate_data_words(self, data, vectorizer):
        pairs = Helper.get_pairs('C:/Users/jakub/Documents/sloucena_id')
        date_now = datetime.now()
        results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        preprocessed_dir = Path("rake")
        preprocessor = Preprocessor()
        extractor = TextExtractor('C:/Users/jakub/Documents/all',
                                  'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
        words = []
        i = 0
        for index, row in data.iterrows():
            # print("processing number: " + str(i))
            values = pairs.get(row['OAI'], None)
            if values is not None:
                uuid = values[0]
                # print(uuid)
                text = extractor.get_text(uuid)
                text_pre = Helper.check_processed(uuid, preprocessed_dir)
                if text_pre is None:
                    print("preprocessing")
                    text_pre = preprocessor.lemmatize(text)
                    # text_pre = ' '.join(text_pre)
                    Helper.save_document(' '.join(text_pre), uuid, preprocessed_dir)
                    print("end preprocessing")
                else:
                    text_pre = text_pre.split(' ')
                text = preprocessor.tokenize(text)
                if len(text) != len(text_pre):
                    print("processing number: " + str(i))
                    print(len(text))
                    print(len(text_pre))
                    print("not equal")
                    raise Exception('texts are not equal')

                title = preprocessor.lemmatize(row['title'])
                keywords = preprocessor.lemmatize(row['keywords'])
                max_lenght = 0
                for w in text:
                    # TODO odstranit neslova
                    new_word = {'uuid': uuid, 'OAI': row['OAI'], 'length': len(w)}
                    if w in title:
                        new_word['in_title'] = True
                    else:
                        new_word['in_title'] = False
                    if w in keywords:
                        new_word['in_keywords'] = True # TODO najst povodne klucove slova a podla toho otagovat
                    else:
                        new_word['in_keywords'] = False
            i += 1


crf = CrfKeywords()
data = pd.read_csv('train2.csv', index_col=0)
data = data.iloc[:1000]
crf.generate_data_words(data, None)
