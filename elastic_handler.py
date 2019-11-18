from elasticsearch import Elasticsearch
import configparser
import os
from elasticsearch_dsl import Search, Q


class ElasticHandler:
    @staticmethod
    def get_config():
        config = configparser.ConfigParser()
        directory = os.path.dirname(os.path.realpath(__file__))
        config.read(directory + '/config.ini')
        return config

    @staticmethod
    def get_environment():
        config = ElasticHandler.get_config()
        environment = config.read_string('elastic', 'environment')
        return environment

    @staticmethod
    def get_index():
        config = ElasticHandler.get_config()
        environment = config.read_string('elastic', 'environment')
        index = config.read_string(environment, 'index')
        return index

    @staticmethod
    def get_test_index():
        config = ElasticHandler.get_config()
        index = config.get('test', 'index')
        return index

    @staticmethod
    def create_document_index(index):

        mappings = {
            "mappings": {
                "_source": {
                    "excludes": ["text"]
                },
                "properties": {
                    "title": {
                        "type": "keyword"
                    },
                    "keywords": {
                        "type": "keyword"
                    },
                    "keywords_generated": {
                        "type": "keyword"
                    },
                    "konspekt": {
                        "type": "nested",
                        "properties": {
                            "category": {
                                "type": "keyword"
                            },
                            "group": {
                                "type": "keyword"
                            },
                            "description": {
                                "type": "keyword"
                            }
                        }
                    },
                    "konspekt_generated": {
                        "type": "nested",
                        "properties": {
                            "category": {
                                "type": "keyword"
                            },
                            "group": {
                                "type": "keyword"
                            },
                            "description": {
                                "type": "keyword"
                            }
                        }
                    },
                    "mdt": {
                        "type": "keyword"
                    },
                    "additional_info": {
                        "type": "text"
                    },
                    "id_001": {
                        "type": "keyword"
                    },
                    "uuid": {
                        "type": "keyword"
                    },
                    "oai": {
                        "type": "keyword"
                    },
                    "isbn": {
                        "type": "keyword"
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "fulltext_analyzer",
                        "term_vector": "with_positions_offsets",
                        "store": True,
                        "index_options": "offsets"
                    },
                    "text_length": {
                        "type": "long"
                    }
                }
            },
            "settings": {
                "index.analyze.max_token_count": 20000,
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "analysis": {
                    "analyzer": {
                        "fulltext_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase"
                            ]
                        }
                    }
                }
            }
        }

        es = Elasticsearch()
        response = es.indices.create(index=index, body=mappings)
        return response

    @staticmethod
    def save_document(index, document):
        es = Elasticsearch()
        # save document
        response = es.index(index=index, body=document)
        return response

    @staticmethod
    def save_konspekt(index, id_elastic, konspekt):
        es = Elasticsearch()
        dicts = []
        for k in konspekt:
            dict_konspekt = {'category': k[0], 'group': k[1], 'description': [2]}
            dicts.append(dict_konspekt)

        # save konspekt
        body = {'doc': {'konspekt_generated': dicts}}
        response = es.update(index=index, id=id_elastic, body=body)
        return response

    @staticmethod
    def save_keywords(index, id_elastic, keywords):
        es = Elasticsearch()
        # save keywords
        body = {'doc': {'keywords_generated': keywords}}
        response = es.update(index=index, id=id_elastic, body=body)
        return response

    @staticmethod
    def term_vectors(index, id_elastic):
        body = {
            "fields": ['text'],
            "positions": True,
            "term_statistics": True,
            "field_statistics": True
        }
        es = Elasticsearch()
        response = es.termvectors(index=index, id=id_elastic, body=body)
        term_vectors = response['term_vectors']['text']['terms']
        doc_count = response['term_vectors']['text']['field_statistics']['doc_count']
        return term_vectors, doc_count

    @staticmethod
    def get_text(index, id_elastic):
        es = Elasticsearch()
        response = es.get(index=index, id=id_elastic)

        return response['_source']['text']

    @staticmethod
    def select_with_mdt(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='mdt')])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_with_keywords_konspekt(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='keywords'),
                    Q('nested', path='konspekt', query=Q('bool', must=[Q('exists', field='konspekt')]))])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_with_keywords_no_konspekt(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='keywords')],
              must_not=[Q('nested', path='konspekt', query=Q('bool', must=[Q('exists', field='konspekt')])),
                        Q('nested', path='konspekt_generated', query=Q('bool', must=[Q('exists',
                                                                                       field='konspekt_generated')]))])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_with_text_konspekt(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='text'),
                    Q('nested', path='konspekt', query=Q('bool', must=[Q('exists', field='konspekt')]))])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_with_text_no_konspekt(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='text')],
              must_not=[Q('nested', path='konspekt', query=Q('bool', must=[Q('exists', field='konspekt')])),
                        Q('nested', path='konspekt_generated', query=Q('bool', must=[Q('exists',
                                                                                       field='konspekt_generated')]))])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_with_text_no_keywords(index):
        es = Elasticsearch()
        q = Q('bool',
              must=[Q('exists', field='text')],
              must_not=[Q('exists', field='keywords'), Q('exists', field='keywords_generated')])
        s = Search(using=es, index=index).query(q)
        s.execute()
        return s

    @staticmethod
    def select_all(index):
        es = Elasticsearch()
        s = Search(using=es, index=index)
        s.execute()
        return s

    @staticmethod
    def get_document(index, id_001):
        es = Elasticsearch()
        exists_query = Search(using=es, index=index).query(Q({"match": {"id_001": id_001}}))
        response = exists_query.execute()
        if len(response.hits) != 0:
            return response.hits[0].to_dict()
        else:
            return None