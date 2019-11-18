import os
import unittest
from data_export import DataExporter
from elastic_handler import ElasticHandler
from elasticsearch import Elasticsearch


class TestDataExporter(unittest.TestCase):
    def test_data_export(self):
        index = ElasticHandler.get_test_index()
        es = Elasticsearch()
        if es.indices.exists(index=index):
            es.indices.delete(index=index)
        response = ElasticHandler.create_document_index(index)
        self.assertEqual(index, response['index'])
        try:
            document = {'title': 'test', 'keywords': ['test', 'test2'],
                        'konspekt': {'category': '1', 'group': '28.3', 'description': 'test'},
                        'mdt': ['mdt1', 'mdt2'], 'id_001': '000000116', 'uuid': 'test_uuid', 'oai': 'test_oai',
                        'isbn': 'test_isbn', 'text': 'text',
                        'konspekt_generated': {'category': '1', 'group': '28.3', 'description': 'test'},
                        'keywords_generated': ['test', 'test2']}
            ElasticHandler.save_document(index, document)
            es.indices.refresh(index=index)
            exporter = DataExporter()
            exporter.add_all_xml('test_data/metadata.xml', 'test_data/export.xml', index)
            with open('test_data/export.xml', 'r') as file:
                text = file.read()
                self.assertEqual('<subfield code="9">1</subfield>' in text, True)
                self.assertEqual('<datafield tag="N072" ind1=" " ind2="7">' in text, True)
                self.assertEqual('<datafield tag="N650" ind1=" " ind2="7">' in text, True)
                self.assertEqual('<subfield code="a">test</subfield>' in text, True)
                self.assertEqual('<subfield code="a">test2</subfield>' in text, True)

        finally:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)
            if os.path.isfile('test_data/export.xml'):
                os.remove('test_data/export.xml')


if __name__ == '__main__':
    unittest.main()
