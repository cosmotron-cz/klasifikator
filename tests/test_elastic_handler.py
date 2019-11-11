import unittest
from elastic_handler import ElasticHandler
from elasticsearch import Elasticsearch


class TestElasticHandler(unittest.TestCase):
    def test_create_index(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_save_document(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn', 'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_save_konspekt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])
            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn', 'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])
            konspekt = [['category1', 'group1', 'description1'], ['category2', 'group2', 'description2']]
            response = ElasticHandler.save_konspekt(index, response['_id'], konspekt)

            self.assertEqual('updated', response['result'])
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_save_keywords(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn', 'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])
            keywords = [['keyword1', 'keyword2', 'keyword3']]
            response = ElasticHandler.save_konspekt(index, response['_id'], keywords)
            self.assertEqual('updated', response['result'])
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_mdt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn', 'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2', 'keywords': ['test', 'test2'],
                         'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2',
                         'text': 'text2'}
            response2 = ElasticHandler.save_document(index, document2)
            es.indices.refresh(index=index)
            self.assertEqual('created', response2['result'])

            hits = ElasticHandler.select_with_mdt(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_keywords_konspekt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn', 'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2',
                         'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2',
                         'text': 'text2'}
            response2 = ElasticHandler.save_document(index, document2)
            es.indices.refresh(index=index)
            self.assertEqual('created', response2['result'])

            hits = ElasticHandler.select_with_keywords_konspekt(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_keywords_no_konspekt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn',
                        'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2', 'keywords': ['test', 'test2'],
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2',
                         'text': 'text2'}
            response2 = ElasticHandler.save_document(index, document2)

            self.assertEqual('created', response2['result'])

            document3 = {'title': 'test3', 'keywords': ['test', 'test2'],
                         'konspekt_generated': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'mdt': ['mdt1', 'mdt2'],
                         'id_001': 'test_id3', 'uuid': 'test_uuid3', 'oai': 'test_oai3', 'isbn': 'test_isbn3',
                         'text': 'text'}
            response3 = ElasticHandler.save_document(index, document3)
            self.assertEqual('created', response3['result'])

            es.indices.refresh(index=index)
            hits = ElasticHandler.select_with_keywords_no_konspekt(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id2', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_text_konspekt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn',
                        'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2', 'keywords': ['test', 'test2'],
                         'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2'}
            response2 = ElasticHandler.save_document(index, document2)
            es.indices.refresh(index=index)
            self.assertEqual('created', response2['result'])

            hits = ElasticHandler.select_with_text_konspekt(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_text_no_konspekt(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn',
                        'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2', 'keywords': ['test', 'test2'],
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2',
                         'text': 'text2'}
            response2 = ElasticHandler.save_document(index, document2)

            self.assertEqual('created', response2['result'])

            document3 = {'title': 'test3', 'keywords': ['test', 'test2'],
                         'konspekt_generated': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'mdt': ['mdt1', 'mdt2'],
                         'id_001': 'test_id3', 'uuid': 'test_uuid3', 'oai': 'test_oai3', 'isbn': 'test_isbn3',
                         'text': 'text'}
            response3 = ElasticHandler.save_document(index, document3)
            self.assertEqual('created', response3['result'])

            es.indices.refresh(index=index)
            hits = ElasticHandler.select_with_text_no_konspekt(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id2', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)

    def test_select_with_text_no_keywords(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])

            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'}, 'mdt': ['mdt1', 'mdt2'],
                        'id_001': 'test_id', 'uuid': 'test_uuid', 'oai': 'test_oai', 'isbn': 'test_isbn',
                        'text': 'text'}
            response = ElasticHandler.save_document(index, document)
            self.assertEqual('created', response['result'])

            document2 = {'title': 'test2',
                         'id_001': 'test_id2', 'uuid': 'test_uuid2', 'oai': 'test_oai2', 'isbn': 'test_isbn2',
                         'text': 'text2'}
            response2 = ElasticHandler.save_document(index, document2)

            self.assertEqual('created', response2['result'])

            document3 = {'title': 'test3', 'keywords_generated': ['test', 'test2'],
                         'konspekt_generated': {'category': '1', 'group': '28.3', 'description': 'test'},
                         'mdt': ['mdt1', 'mdt2'],
                         'id_001': 'test_id3', 'uuid': 'test_uuid3', 'oai': 'test_oai3', 'isbn': 'test_isbn3',
                         'text': 'text'}
            response3 = ElasticHandler.save_document(index, document3)
            self.assertEqual('created', response3['result'])

            es.indices.refresh(index=index)
            hits = ElasticHandler.select_with_text_no_keywords(index)
            i = 0
            for hit in hits:
                hit_dict = hit.to_dict()
                self.assertEqual('test_id2', hit_dict['id_001'])
                i += 1
            self.assertEqual(1, i)
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)


if __name__ == '__main__':
    unittest.main()
