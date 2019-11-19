import unittest
from keywords_generator import KeywordsGenerator
from elastic_handler import ElasticHandler
from elasticsearch import Elasticsearch
from preprocessor import Preprocessor


class TestKeywordsGenerator(unittest.TestCase):
    def test_generate_keywords(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        if es.indices.exists(index=index):
            es.indices.delete(index=index)
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])
            text = 'Jak vlastně vypadají ony balónky?. Ptají se často lidé. Inu jak by vypadaly - jako běžné pouťové ' \
                   'balónky střední velikosti, tak akorát nafouknuté. Červený se vedle modrého a zeleného zdá trochu ' \
                   'menší, ale to je nejspíš jen optický klam, a i kdyby byl skutečně o něco málo menší, tak vážně.'
            preprocessor = Preprocessor()
            text = preprocessor.remove_stop_words(text)
            text = preprocessor.lemmatize(text)
            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn',
                        'text': ' '.join(text)}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            generator = KeywordsGenerator()
            keywords = generator.generate_keywords_elastic(index, response['_id'])
            for word in keywords:
                self.assertEqual(isinstance(word, str), True)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)


if __name__ == '__main__':
    unittest.main()
