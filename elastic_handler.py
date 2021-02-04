from elasticsearch import Elasticsearch, ElasticsearchException
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
        environment = config.get('elastic', 'environment')
        return environment

    @staticmethod
    def get_index():
        config = ElasticHandler.get_config()
        environment = config.get('elastic', 'environment')
        index = config.get(environment, 'index')
        return index

    @staticmethod
    def get_test_index():
        config = ElasticHandler.get_config()
        index = config.get('test', 'index')
        return index

    @staticmethod
    def get_text_index(doc_index):
        parts = doc_index.split('_', 1)
        return "fulltext_" + parts[1]

    @staticmethod
    def create_document_index(index):

        mappings = {
            "mappings": {
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

        mapping_text = {
            "mappings": {
                "properties": {
                    "id_document": {
                        "type": "keyword"
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "fulltext_analyzer",
                        "term_vector": "with_positions_offsets",
                        "store": True,
                        "index_options": "offsets"
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
        if response['acknowledged'] is False:
            raise Exception("Couldn't create index: " + str(index))
        text_index = ElasticHandler.get_text_index(index)
        response_text = es.indices.create(index=text_index, body=mapping_text)
        if response_text['acknowledged'] is False:
            raise Exception("Couldn't create index: " + str(index))
        return response

    @staticmethod
    def save_document(index, document):
        text = document.get('text', None)
        if text is not None:
            del document['text']
        es = Elasticsearch()
        # save document
        response = es.index(index=index, body=document)
        if response['result'] != 'created':
            raise Exception("Couldn't save document")

        if text is not None:
            text_index = ElasticHandler.get_text_index(index)
            body = {"text": text, "id_document": response['_id']}
            response_text = es.index(index=text_index, body=body, request_timeout=60)
            if response_text['result'] != 'created':
                raise Exception("Couldn't save document text")
        return response

    @staticmethod
    def save_konspekt(index, id_elastic, konspekt):
        es = Elasticsearch()
        dicts = []
        for k in konspekt:
            if k is None:
                continue
            dict_konspekt = {'category': k['category'], 'group': k['group'], 'description': k['description']}
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
        text_index = ElasticHandler.get_text_index(index)
        text_id = ElasticHandler.get_text_id(index, id_elastic)
        if text_id is None:
            return None, None
        body = {
            "fields": ['text'],
            "positions": True,
            "term_statistics": True,
            "field_statistics": True
        }
        es = Elasticsearch()
        response = es.termvectors(index=text_index, id=text_id, body=body)
        try:
            term_vectors = response['term_vectors']['text']['terms']
            doc_count = response['term_vectors']['text']['field_statistics']['doc_count']
        except KeyError:
            return None, None
        return term_vectors, doc_count

    @staticmethod
    def term_vectors_keywords(index, id_elastic):
        text_index = ElasticHandler.get_text_index(index)
        text_id = ElasticHandler.get_text_id(index, id_elastic)
        if text_id is None:
            return None
        body = {
            "fields": ["text"],
            "term_statistics": True,
            "field_statistics": True,
            "positions": False,
            "offsets": False,
            "filter": {
                "max_num_terms": 30,
                "min_term_freq": 1,
                "min_doc_freq": 1
            }
        }
        es = Elasticsearch()
        response = es.termvectors(index=text_index, id=text_id, body=body)
        try:
            term_vectors = response['term_vectors']['text']['terms']
        except KeyError:
            return None
        return term_vectors

    @staticmethod
    def get_text(index, id_elastic):
        text_index = ElasticHandler.get_text_index(index)
        text_id = ElasticHandler.get_text_id(index, id_elastic)
        es = Elasticsearch()
        response = es.get(index=text_index, id=text_id)

        return response['_source']['text']

    @staticmethod
    def get_text_id(index, id_elastic):
        es = Elasticsearch()
        text_index = ElasticHandler.get_text_index(index)
        exists_query = Search(using=es, index=text_index).query(Q({"match": {"id_document": id_elastic}}))
        response = exists_query.execute()
        if len(response.hits) != 0:
            return response.hits[0].meta.id
        else:
            return None

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
    def select_all(index):
        es = Elasticsearch(timeout=60)
        s = Search(using=es, index=index)
        s = s.params(scroll='5h', request_timeout=100)
        s.execute()
        return s.scan()

    @staticmethod
    def get_document(index, id_001):
        es = Elasticsearch()
        exists_query = Search(using=es, index=index).query(Q({"match": {"id_001": id_001}}))
        response = exists_query.execute()
        if len(response.hits) != 0:
            return response.hits[0].to_dict()
        else:
            return None

    @staticmethod
    def remove_index(index):
        es = Elasticsearch()
        if es.indices.exists(index=index):
            es.indices.delete(index=index)
        text_index = ElasticHandler.get_text_index(index)
        if es.indices.exists(index=text_index):
            es.indices.delete(index=text_index)

    @staticmethod
    def refresh(index):
        es = Elasticsearch()
        es.indices.refresh(index=index)
        text_index = ElasticHandler.get_text_index(index)
        es.indices.refresh(index=text_index)

    @staticmethod
    def create_planned_classification(data, model, note, date, email, index=None):
        if index is None or index == "":
            index = "planned_classification"
        es = Elasticsearch()
        doc = {
            'data': data,
            'model': model,
            'note': note,
            'date': date,
            'email': email,
            'status': 'Planned',
            'error': '',
            'export': ''
        }

        res = es.index(index=index, body=doc)

    @staticmethod
    def update_planned_classification(class_id, data, model, note, date, email, status, error, export, index=None):
        if index is None or index == "":
            index = "planned_classification"
        es = Elasticsearch()

        doc = {'doc': {}}
        if data is not None:
            doc['doc']['data'] = data

        if model is not None:
            doc['doc']['model'] = model

        if note is not None:
            doc['doc']['note'] = note

        if date is not None:
            doc['doc']['date'] = date

        if email is not None:
            doc['doc']['email'] = email

        if status is not None:
            doc['doc']['status'] = status

        if error is not None:
            doc['doc']['error'] = error

        if export is not None:
            doc['doc']['export'] = export

        res = es.update(index=index, id=class_id, body=doc)

    @staticmethod
    def delete_planned_classification(class_id, index=None):
        if index is None or index == "":
            index = "planned_classification"
        es = Elasticsearch()

        res = es.delete(index=index, id=class_id)

    @staticmethod
    def get_planned_classifications(index=None):
        if index is None or index == "":
            index = "planned_classification"
        es = Elasticsearch()

        res = es.search(index=index, body={"query": {"match_all": {}}})
        data = []
        for doc in res['hits']['hits']:
            training = {'id': doc['_id'],
                        'data': doc['_source'].get('data', ''),
                        'model': doc['_source'].get('model', ''),
                        'note': doc['_source'].get('note', ''),
                        'date': doc['_source'].get('date', ''),
                        'email': doc['_source'].get('email', ''),
                        'status': doc['_source'].get('status', ''),
                        'error': doc['_source'].get('error', ''),
                        'export': doc['_source'].get('export', '')}
            data.append(training)
        return data

    @staticmethod
    def get_unstarted_classifications(index=None):
        if index is None or index == "":
            index = "planned_classification"
        es = Elasticsearch()

        res = es.search(index=index, body={"query": {"match": {"status": "Planned"}}})
        data = []
        for doc in res['hits']['hits']:
            classification = {'id': doc['_id'],
                              'data': doc['_source'].get('data', ''),
                              'model': doc['_source'].get('model', ''),
                              'note': doc['_source'].get('note', ''),
                              'date': doc['_source'].get('date', ''),
                              'email': doc['_source'].get('email', ''),
                              'status': doc['_source'].get('status', ''),
                              'error': doc['_source'].get('error', ''),
                              'export': doc['_source'].get('export', '')}
            data.append(classification)
        return data

    @staticmethod
    def create_planned_training(data, note, date, email, index=None):
        if index is None or index == "":
            index = "planned_training"
        es = Elasticsearch()
        doc = {
            'data': data,
            'note': note,
            'date': date,
            'email': email,
            'status': 'Planned',
            'error': '',
            'model': ''
        }

        res = es.index(index=index, body=doc)

    @staticmethod
    def update_planned_training(training_id, data, note, date, email, status, error, model, index=None):
        if index is None or index == "":
            index = "planned_training"
        es = Elasticsearch()
        doc = {'doc': {}}
        if data is not None:
            doc['doc']['data'] = data

        if note is not None:
            doc['doc']['note'] = note

        if date is not None:
            doc['doc']['date'] = date

        if email is not None:
            doc['doc']['email'] = email

        if status is not None:
            doc['doc']['status'] = status

        if error is not None:
            doc['doc']['error'] = error

        if model is not None:
            doc['doc']['model'] = model

        res = es.update(index=index, id=training_id, body=doc)
        print(res)

    @staticmethod
    def delete_planned_training(training_id, index=None):
        if index is None or index == "":
            index = "planned_training"
        es = Elasticsearch()

        res = es.delete(index=index, id=training_id)

    @staticmethod
    def get_planned_trainings(index=None):
        if index is None or index == "":
            index = "planned_training"
        es = Elasticsearch()

        res = es.search(index=index, body={"query": {"match_all": {}}})
        data = []
        for doc in res['hits']['hits']:
            training = {'id': doc['_id'],
                        'data': doc['_source'].get('data', ''),
                        'note': doc['_source'].get('note', ''),
                        'date': doc['_source'].get('date', ''),
                        'email': doc['_source'].get('email', ''),
                        'status': doc['_source'].get('status', ''),
                        'error': doc['_source'].get('error', ''),
                        'model': doc['_source'].get('model', '')}
            data.append(training)
        return data

    @staticmethod
    def get_unstarted_trainings(index=None):
        if index is None or index == "":
            index = "planned_training"
        es = Elasticsearch()

        res = es.search(index=index, body={"query": {"match": {"status": "Planned"}}})
        data = []
        for doc in res['hits']['hits']:
            training = {'id': doc['_id'],
                        'data': doc['_source'].get('data', ''),
                        'note': doc['_source'].get('note', ''),
                        'date': doc['_source'].get('date', ''),
                        'email': doc['_source'].get('email', ''),
                        'status': doc['_source'].get('status', ''),
                        'error': doc['_source'].get('error', ''),
                        'model': doc['_source'].get('model', '')}
            data.append(training)
        return data
