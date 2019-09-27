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
from multi_rake import Rake
from pathlib import Path
from lxml import etree
from io import StringIO

class KeywordsGeneratorTfidf:
    def fit_from_elastic(self, index):
        pairs = self.get_pairs()
        all_files = []
        directory = 'C:/Users/jakub/Documents/all'
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
                              'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages', uuids=uuids)

        vectorizer = Vectorizer(vectorizer='tfidf', ngram=1)
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
        directory = Path('C:/Users/jakub/Documents/all')
        extractor = TextExtractor(directory,
                                  'C:/Users/jakub/Documents/sorted_pages_zip/sorted_pages')

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
                print("end preprocessing")
                kw = self.extract_keywords(vectorizer, text)
                result = ' '.join(kw)
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
        keywords = self.extract_topn_from_vector(feature_names, sorted_items, 10)
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
        dataframe.to_csv(results_dir / name)

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
        print("recall: " + str(recall_1/number))
        print("preccision: " + str(precision_1/number))
        print("f1: " + str(f1_1/number))

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





index = "keyword_czech"
kg = KeywordsGeneratorTfidf()
# # kg.fit_from_elastic(index)
test = pd.read_csv('rake\\test_contents.csv', index_col=0)
kg.evaluate_keywords(test)
# test = test.iloc[:1000]
# vectorizer = Vectorizer(load_vec='vectorizers\\pre_contents_1000_bi.pickle')
# kg.tfidf_keywords2(test, vectorizer)
# kg.rake_keywords(test, True)
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