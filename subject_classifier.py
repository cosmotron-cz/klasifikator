import os
import argparse
import sys
from pathlib import Path

from scipy.sparse import csr_matrix
import numpy as np

from match_konspect import MatchKonspekt
from elastic_handler import ElasticHandler
from preprocessor import Preprocessor
from keywords_generator import KeywordsGenerator
from helper.helper import Helper
from data_import import DataImporter
from data_export import DataExporter
from elasticsearch import Elasticsearch


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

    def import_data(self, path):
        index = ElasticHandler.get_index()
        ElasticHandler.create_document_index(index)
        importer = DataImporter()
        importer.import_data(path, index)
        ElasticHandler.refresh(index)

    def classify_documents(self):
        print("classify documents start")
        index = ElasticHandler.get_index()
        all_documents = ElasticHandler.select_all(index)
        for document in all_documents:
            id_elastic = document.meta.id
            document = document.to_dict()
            generated = False

            mdt = document.get('mdt', None)
            if mdt is not None:
                konspekt_generated = self.match_konspekt.find_and_choose(mdt)
                ElasticHandler.save_konspekt(index, id_elastic, konspekt_generated)
                generated = True

            keywords = document.get('keywords', None)
            if generated is False and keywords is not None:
                keywords = ' '.join(keywords)
                keywords = self.preprocessor.remove_stop_words(keywords)
                keywords = self.preprocessor.lemmatize(keywords)
                category, group = self.classifier_keywords.classify(keywords)
                description = self.rules.get(group, "")
                konspekt_generated = {"category": category, "group": group, "description": description}
                ElasticHandler.save_konspekt(index, id_elastic, konspekt_generated)
                generated = True

            if generated is False:
                term_vectors, doc_count = ElasticHandler.term_vectors(index, id_elastic)
                if term_vectors is not None and doc_count is not None:
                    category, group = self.classifier_fulltext.classify(term_vectors, document['text_length'],
                                                                        doc_count)
                    description = self.rules.get(group, "")
                    konspekt_generated = {"category": category, "group": group, "description": description}
                    ElasticHandler.save_konspekt(index, id_elastic, konspekt_generated)
                    generated = True

            if keywords is None:
                keywords = self.keyword_generator.generate_keywords_elastic(index, id_elastic)
                ElasticHandler.save_keywords(index, id_elastic, keywords)
        ElasticHandler.refresh(index)
        print("classify documents end")

    def export_data(self, path_from, path_to=None):
        print("export data start")
        if path_to is None:
            path_to = 'export.xml'
        index = ElasticHandler.get_index()
        path = Path(path_from)
        xml_path = path / 'metadata.xml'
        exporter = DataExporter()
        exporter.add_all_xml(xml_path, path_to, index)
        print("export data end")


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

    def classify(self, term_vectors, text_length, doc_count):
        tfidf_vector = []
        for word in self.dictionary:
            term = term_vectors.get(word, None)
            if term is None:
                tfidf_vector.append(0.0)
            else:
                tfidf = self.tfidf(term['term_freq'], term['doc_freq'], doc_count, text_length)
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

    def tfidf(self, term_freq, doc_freq, doc_count, sum):
        inverse_doc_freq = np.log(doc_count / doc_freq)
        return term_freq / sum * inverse_doc_freq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nástroj pre vecnú klasifikáciu dokumentov")
    parser.add_argument("--directory", help="Adresár s potrebnými dátami")
    parser.add_argument("--action", help="Akcia, ktorá sa vykoná. Možné akcie: \n"
                                         "all - vykoná celý proces klasifikácie\n"
                                         "import - import dát do elastiku\n"
                                         "classify - klasifikovanie dát uložených v elastiku\n"
                                         "export - export dát z elastiku do xml\n"
                                         "remove - odstranenie dat z elastiku")
    parser.add_argument("--export_to", help="Súbor kam sa exportujú dáta")
    args = parser.parse_args()
    if args.action is None or args.action == 'all':
        if args.directory is None or args.export_to is None:
            print("Pre vykonanie akcie all je potrebné zadať parametre --directory a --export_to")
            sys.exit(2)

        classifier = SubjectClassifier()
        classifier.import_data(args.directory)
        classifier.classify_documents()
        classifier.export_data(args.directory, args.export_to)
    elif args.action == "import":
        if args.directory is None:
            print("Pre vykonanie akcie import je potrebné zadať parameter --directory")
            sys.exit(2)
        classifier = SubjectClassifier()
        classifier.import_data(args.directory)
    elif args.action == "classify":
        classifier = SubjectClassifier()
        classifier.classify_documents()
    elif args.action == "export":
        if args.directory is None or args.export_to is None:
            print("Pre vykonanie akcie export je potrebné zadať parametre --directory a --export_to")
            sys.exit(2)
        classifier = SubjectClassifier()
        classifier.export_data(args.directory, args.export_to)
    elif args.action == "remove":
        index = ElasticHandler.get_index()
        ElasticHandler.remove_index(index)
    else:
        print("Neznáma akcia: " + str(args.action))
        sys.exit(2)
