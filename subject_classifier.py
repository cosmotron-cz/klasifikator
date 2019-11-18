import os

from scipy.sparse import csr_matrix
import numpy as np

from match_konspect import MatchKonspekt
from elastic_handler import ElasticHandler
from preprocessor import Preprocessor
from keywords_generator import KeywordsGenerator
from helper.helper import Helper


class SubjectClassifier:
    def __init__(self):

        self.match_konspekt = MatchKonspekt()
        self.elastic_handler = ElasticHandler()
        self.preprocessor = Preprocessor()
        self.classifier_fulltext = ClassifierFulltext(self.preprocessor)
        self.classifier_keywords = ClassifierKeywords(self.preprocessor)
        self.keyword_generator = KeywordsGenerator()
        self.rules = {}
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        k = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "/rules.txt", "r", encoding="utf8") as file:
            for line in file:
                parts = line.split("$$")
                if "b1" == parts[2]:
                    k += 1
                    continue

                pattern = parts[1][1:]
                desc = parts[3][1:]
                new_dict = {"category": k, "description": desc, "original": ""}
                self.rules[pattern] = new_dict

    def classify_document(self, metadata, fulltext):
        # prepossessing fulltext and save document to elastic
        fulltext_pre = self.preprocessor.remove_stop_words(fulltext)
        fulltext_pre = self.preprocessor.lemmatize(fulltext_pre)
        id_elastic = self.elastic_handler.save_document(metadata, fulltext, fulltext_pre)
        if id_elastic is None:
            raise Exception('Could not save document')

        konspekt = None
        keywords = None
        if metadata['mdt'] is not None:
            # generate konspekt from mdt
            konspekt = self.match_konspekt.find_and_choose(metadata['mdt'])  # TODO skontolovat

        if konspekt is None and metadata.get('keywords', None) is not None:
            # generate konspekt from keywords
            konspekt = self.classifier_keywords.classify_elastic(id_elastic)  # TODO skontolovat

        if konspekt is None:
            # generate konspekt from fulltext
            konspekt = self.classifier_fulltext.classify_elastic(id_elastic)  # TODO skontolovat

        if metadata.get('keywords', None) is None:
            # generate keywords from fulltext
            keywords = self.keyword_generator.generate_kewords_elastic(id_elastic)  # TODO skontolovat

        return konspekt, keywords


class ClassifierKeywords:
    def __init__(self, preprocessor):
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        self.category_classifier = Helper.load_model(self.script_directory + '/models/keywords/category.pickle')
        self.groups = {}
        for i in range(1, 27):
            model = Helper.load_model(self.script_directory + '/models/keywords/groups_' + str(i) + '.pickle')
            self.groups[str(i)] = model
        self.label_encoder = Helper.load_model(self.script_directory + '/models/keywords/groups_labels.pickle')
        self.tfidf = Helper.load_model(self.script_directory + '/models/keywords/tfidf.pickle')
        self.preprocessor = preprocessor


    def classify(self, keywords):
        text = ""
        for word in keywords:
            text += "" + word

        text = self.preprocessor.remove_stop_words(text)
        text = self.preprocessor.lemmatize(text)
        matrix = self.tfidf.transform([' '.join(text)])
        category = self.category_classifier.predict(matrix)[0]
        group = self.groups[str(category)].predict(matrix)
        group = self.label_encoder.inverse_transform(group)[0]

        return category, group


class ClassifierFulltext:
    def __init__(self, preprocessor):
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        self.category_classifier = Helper.load_model(self.script_directory + '/models/fulltext/category.pickle')
        self.groups = {}
        for i in range(1, 27):
            model = Helper.load_model(self.script_directory + '/models/fulltext/groups_' + str(i) + '.pickle')
            self.groups[str(i)] = model
        self.label_encoder = Helper.load_model(self.script_directory + '/models/fulltext/groups_labels.pickle')
        self.preprocessor = preprocessor
        self.dictionary = self.prepare_dictionary(self.script_directory + '/dictionary.txt')

    def classify(self, term_vectors, word_count, doc_count):
        tfidf_vector = []
        for word in self.dictionary:
            term = term_vectors.get(word, None)
            if term is None:
                tfidf_vector.append(0.0)
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, word_count)
                tfidf_vector.append(tfidf)

        matrix = csr_matrix(tfidf_vector)

        category = self.category_classifier.predict(matrix)[0]
        group = self.groups[str(category)].predict(matrix)
        group = self.label_encoder.inverse_transform(group)[0]

        return category, group

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

    def tfidf(self, term_freq, doc_freq,  doc_count, sum):
        inverse_doc_freq = np.log(doc_count/doc_freq)
        return term_freq / sum * inverse_doc_freq
