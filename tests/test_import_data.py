import unittest
from data_import import DataImporter
from elastic_handler import ElasticHandler
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q


class TestDataImport(unittest.TestCase):
    def test_data_import(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        if es.indices.exists(index=index):
            es.indices.delete(index=index)
        response = ElasticHandler.create_document_index(index)
        try:
            self.assertEqual(index, response['index'])
            path = 'test_data'
            DataImporter.import_data(path, index)
            es.indices.refresh(index=index)
            exists_query = Search(using=es, index=index).query(Q({"match": {"id_001": "000000116"}}))
            response = exists_query.execute()
            self.assertEqual(1, len(response.hits))
        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)


if __name__ == '__main__':
    unittest.main()
