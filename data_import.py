import xmltodict
import xml.etree.ElementTree as etree
from elasticsearch import Elasticsearch
from elasticsearch import exceptions


class DataImporter:

    # premiestni nazvy poli marc zaznamov z value do key casti dictionary
    # zmeni "datafield": {"@tag": "072", "subfields": {"@code":"aaa", "#text":"text"}} -> "072": {"aaa":"text"}
    # opakovane polia a podpolia da do zoznamu

    @staticmethod
    def move_tag_names(record):

        for controlfield in record.get('controlfield', []):
            tag = controlfield.get('@tag', None)
            if tag is not None:
                record[str(tag)] = controlfield.get('#text', None)
        record.pop('controlfield', None)

        for datafield in record.get('datafield', []):
            if isinstance(datafield.get('subfield', None), list):
                for subfield in datafield.get('subfield', []):
                    code = subfield.get('@code', None)
                    if code is not None:
                        existing_subfield = datafield.get(str(code), None)
                        new_subfield = subfield.get('#text', None)
                        if existing_subfield is not None:
                            if isinstance(existing_subfield, list):
                                existing_subfield.append(new_subfield)
                            else:
                                datafield[str(code)] = [existing_subfield, new_subfield]
                        else:
                            datafield[str(code)] = new_subfield
            else:
                subfield = datafield.get('subfield', None)
                code = subfield.get('@code', None)
                if code is not None:
                    datafield[str(code)] = subfield.get('#text', None)
            datafield.pop('subfield', None)
            tag = datafield.get('@tag', None)
            if tag is not None:
                existing_datafield = record.get(str(tag), None)
                if existing_datafield is not None:
                    if isinstance(existing_datafield, list):
                        existing_datafield.append(datafield)
                    else:
                        record[str(tag)] = [existing_datafield, datafield]
                else:
                    record[str(tag)] = datafield
            datafield.pop('@tag', None)
        record.pop('datafield', None)

    # importuje metadata do elastiku

    @staticmethod
    def import_metadata(path, index):

        es = Elasticsearch()

        number = 0
        error_log = []
        for event, elem in etree.iterparse(path, events=('end', 'start-ns')):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    number += 1
                    result = None
                    print("processing record number " + str(number))
                    try:
                        xmlstr = etree.tostring(elem, encoding='unicode', method='xml')
                        elem.clear()
                        result = xmltodict.parse(xmlstr)
                        result = result['record']  # odstranenie record tagu
                        DataImporter.move_tag_names(result)
                        if result.get('072', None) is None:
                            continue
                        if result.get('080', None) is None:
                            continue
                        if result.get('OAI', None) is None:
                            continue
                        if not DataImporter.is_in_language_dict(result, 'cze'):
                            continue
                    except Exception as error:
                        print("exception during proccesing record number: " + str(number))
                        error_log.append("exception during proccesing record number: " + str(number))
                        print(error)
                        error_log.append(error)
                        result = None
                        pass
                    if result is not None:
                        try:
                            es.index(index=index, doc_type="record", body=result)
                            es.indices.refresh(index=index)
                            print("saved record number " + str(number))
                        except exceptions.ElasticsearchException as elasticerror:
                            print("failed to save record number: " + str(number))
                            error_log.append("failed to save record number: " + str(number))
                            print(elasticerror)
                            error_log.append(str(elasticerror))
                            pass
        print(error_log)

    # importuje metada do elastiku
    # uklada az od zadaneho cisla

    @staticmethod
    def import_part_of_data(path, index, position_from):

        es = Elasticsearch()

        number = 0
        error_log = []
        for event, elem in etree.iterparse(path, events=('end', 'start-ns')):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    number += 1
                    if number < position_from:
                        elem.clear()
                        continue
                    result = None
                    print("processing record number " + str(number))
                    try:
                        xmlstr = etree.tostring(elem, encoding='unicode', method='xml')
                        elem.clear()
                        result = xmltodict.parse(xmlstr)
                        result = result['record']  # odstranenie record tagu
                        DataImporter.move_tag_names(result)
                        if result.get('072', None) is None:
                            continue
                        if result.get('080', None) is None:
                            continue
                        if result.get('OAI', None) is None:
                            continue
                    except Exception as error:
                        print("exception during proccesing record number: " + str(number))
                        error_log.append("exception during proccesing record number: " + str(number))
                        print(error)
                        error_log.append(error)
                        result = None
                        pass
                    if result is not None:
                        try:
                            es.index(index=index, doc_type="record", body=result)
                            es.indices.refresh(index=index)
                            print("saved record number " + str(number))
                        except exceptions.ElasticsearchException as elasticerror:
                            print("failed to save record number: " + str(number))
                            error_log.append("failed to save record number: " + str(number))
                            print(elasticerror)
                            error_log.append(elasticerror)
                            pass
        print(error_log)

    @staticmethod
    def is_in_language_dict(record, language):
        field_008 = record.get('008', None)
        field_041 = record.get('041', None)
        if field_008 is not None:
            if field_008.lower()[35:38] == language:
                return True
        if field_041 is not None:
            if isinstance(field_041, list):
                for field in field_041:
                    for subfield_a in field.get('a', []):
                        if subfield_a.lower() == language:
                            return True
            else:
                for subfield_a in field_041.get('a', []):
                    if subfield_a.lower() == language:
                        return True
        return False

    @staticmethod
    def count_czech_not_tagged(path):
        number_of_czech_not_taged = 0
        number_with_080 = 0
        number_with_multiple_080 = 0
        error_log = []
        for event, elem in etree.iterparse(path, events=('end', 'start-ns')):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    try:
                        xmlstr = etree.tostring(elem, encoding='unicode', method='xml')
                        elem.clear()
                        result = xmltodict.parse(xmlstr)
                        result = result['record']  # odstranenie record tagu
                        DataImporter.move_tag_names(result)
                        if result.get('072', None) is not None:
                            continue
                        if result.get('OAI', None) is None:
                            continue
                        field_080 = result.get('080', None)
                        if DataImporter.is_in_language_dict(result, "cze"):
                            number_of_czech_not_taged += 1
                        else:
                            continue
                        if field_080 is None:
                            continue
                        else:
                            number_with_080 += 1
                            if isinstance(field_080, list) or isinstance(field_080.get('a', None), list):
                                number_with_multiple_080 += 1
                    except Exception as error:
                        print(error)
                        error_log.append(str(error))
                        pass
        print("number of not tagged in czech: " + str(number_of_czech_not_taged))
        print("number of not tagged in czech with one 080: " + str(number_with_080))
        print("number of not tagged in czech with multiple 080: " + str(number_with_multiple_080))
        print(error_log)

index = 'records_nkp_filtered'
es = Elasticsearch()
# es.indices.delete(index=index)
request_body = {
    "settings" : {
        "number_of_shards": 5,
        "number_of_replicas": 1,
        "index.mapping.total_fields.limit": 3000
    }
}
es.indices.create(index=index, body=request_body)

path = 'C:\\Users\\jakub\\Documents\\metadata_nkp.xml'
di = DataImporter()
di.import_metadata(path, index)
# di.import_part_of_data(path, index, 1088969)
# di.count_czech_not_tagged(path)

