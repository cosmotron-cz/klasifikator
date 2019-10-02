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
import re
from os import listdir
from os.path import isfile, join

class CrfKeywords:
    def generate_data_words(self, data, vectorizer, vectorizer_pre):
        es = Elasticsearch()
        pairs = Helper.get_pairs('C:/Users/jakub/Documents/sloucena_id')
        date_now = datetime.now()
        results_dir = date_now.strftime('%Y_%m_%d_%H_%M')
        preprocessed_dir = Path("rake")
        preprocessor = Preprocessor()
        extractor = TextExtractor('C:/Users/jakub/Documents/all',
                                  'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
        feature_names = vectorizer.vectorizer.get_feature_names()
        feature_names_pre = vectorizer_pre.vectorizer.get_feature_names()

        tagged_fn = preprocessor.pos_tag(feature_names)
        tagged_fn = self.dict_taggs(tagged_fn, feature_names)
        tagged_fn_pre = preprocessor.pos_tag(feature_names_pre)
        tagged_fn_pre = self.dict_taggs(tagged_fn_pre, feature_names_pre)
        words = []
        i = 0
        for indxrow, row in data.iterrows():
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

                title = preprocessor.tokenize(row['title'])
                keywords = preprocessor.tokenize(row['keywords'])
                title_pre = preprocessor.lemmatize(row['title'])
                keywords_pre = preprocessor.lemmatize(row['keywords'])
                # zatial klucove slova iba ako samostante slova
                # q = Q({"match": {"001": row['001']}})
                # s = Search(using=es, index="keyword_czech").query(q)
                # s.execute()
                # for hit in s:
                #     try:
                #         hit_dict = hit.to_dict()
                #         field_650 = hit_dict['650']
                #         if isinstance(field_650, list):
                #             for field in field_650:
                #                 keywords.append(field.get('a', ""))
                #                 pom = preprocessor.lemmatize(field.get('a', ""))
                #                 keywords_pre.append(' '.join(pom))
                #         else:
                #             keywords.append(field_650.get('a', ""))
                #             pom = preprocessor.lemmatize(field_650.get('a', ""))
                #             keywords_pre.append(' '.join(pom))
                #         break
                #     except KeyError:
                #         raise Exception('keyerror while getting keywords')


                max_lenght = 0
                max_lenght_pre = 0
                text2 = []
                text_pre2 = []
                for indx, w in enumerate(text_pre):
                    match = re.search(r"\A[\W]+\Z", w)
                    if match is not None:
                        continue
                    if len(w) > max_lenght_pre:
                        max_lenght_pre = len(w)
                    if len(text[indx]) > max_lenght:
                        max_lenght = len(text[indx])
                    text_pre2.append(w)
                    text2.append(text[indx])

                text = text2
                text_pre = text_pre2
                tags = self.tag_text(text, keywords)
                tags_pre = self.tag_text(text_pre, keywords_pre)
                vector = vectorizer.transform([' '.join(text)])
                tfidf_dict = self.dict_from_vector(vector.toarray(), feature_names)
                vector_pre = vectorizer_pre.transform([' '.join(text_pre)])
                tfidf_dict_pre = self.dict_from_vector(vector_pre.toarray(), feature_names_pre)

                for indx, w in enumerate(text_pre):
                    tfidf = tfidf_dict.get(text[indx], None)
                    if tfidf is None:
                        tfidf = 0.0
                    pos = tagged_fn.get(text[indx], None)
                    if pos is None:
                        pos = 'X@'
                    if tfidf is None:
                        tfidf = 0.0
                    new_word = {'word': text[indx], 'uuid': uuid, 'OAI': row['OAI'],
                                'length': len(w), 'tag': tags[indx], 'tfidf': tfidf, 'pos': pos}
                    if w in title:
                        new_word['in_title'] = True
                    else:
                        new_word['in_title'] = False
                    if indx > 0:
                        new_word['before'] = text[indx-1]
                    else:
                        new_word['before'] = ''
                    if indx <= len(text) - 2:
                        new_word['after'] = text[indx+1]
                    else:
                        new_word['after'] = ''
            i += 1

    def tag_text(self, text, keywords):
        tags = []

        i = 0
        while i < len(text):
            found = False
            for key in keywords:
                len_keyword = len(key.split(' '))
                if i+len_keyword >= len(text):
                    continue
                part = text[i:i+len_keyword]
                part = ' '.join(part)
                if part == key:
                    if len_keyword == 1:
                        tags.append('k')
                    else:
                        tags.append('b')
                        for p in text[i+1:i+len_keyword]:
                            tags.append('p')
                    i += len_keyword
                    found = True
                    break
            if not found:
                if text[i] in Helper.stop_words:
                    tags.append('s')
                else:
                    tags.append('w')
                i += 1
        return tags

    def dict_from_vector(self, vector, feature_names):
        tfidf_dict = {}
        for indx, number in enumerate(vector[0]):
            if number > 0.0:
                tfidf_dict[feature_names[indx]] = number
        return tfidf_dict

    def dict_taggs(self, tags, feature_names):
        tagged_dict = {}
        for i, feature in enumerate(feature_names):
            tagged_dict[feature] = tags[i]
        return tagged_dict

    def create_vectorizer_pre(self):
        mypath = "C:\\Users\\jakub\\PycharmProjects\\klasifikator\\rake\\processed"
        onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.txt')]
        vectorizer = Vectorizer(input='filename')
        vectorizer.fit(onlyfiles)
        path = Helper.create_results_dir()
        vectorizer.save(path)

    def create_vectorizer(self, data):
        te = TextExtractor('C:/Users/jakub/Documents/all',
                           'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')
        texts = []
        pairs = Helper.get_pairs('C:/Users/jakub/Documents/sloucena_id')
        for indx, row in data.iterrows():
            values = pairs.get(row['OAI'])
            if values is not None:
                text = te.get_text(values[0])
                texts.append(text)
        vectorizer = Vectorizer()
        vectorizer.fit(texts)
        path = Helper.create_results_dir()
        vectorizer.save(path)


vectorizer_pre = Vectorizer(load_vec='2019_10_01_15_02\\unigram_tfidf_with_stop_words_1000.pickle')
vectorizer_pre.vectorizer.input = 'content'
vectorizer = Vectorizer(load_vec='2019_10_01_15_23\\unigram_tfidf_nopre_1000.pickle')
crf = CrfKeywords()
data = pd.read_csv('train2.csv', index_col=0)
data = data.iloc[:1000]
crf.generate_data_words(data, vectorizer, vectorizer_pre)


