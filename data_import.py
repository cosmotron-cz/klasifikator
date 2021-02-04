# coding=utf-8
import io

import xmltodict
import xml.etree.ElementTree as etree
from elasticsearch import Elasticsearch
from elasticsearch import exceptions
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.utils import AttrDict
import os
import tarfile
from helper.helper import Helper
from helper.text_extractor import TextExtractor, TextExtractorPre
from pathlib import Path
import re
import time
from preprocessor import Preprocessor
from elastic_handler import ElasticHandler


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
    def import_data(path, index):
        print("import data start")
        path = Path(path)
        pairs = Helper.get_pairs(path / 'sloucena_id')
        te_pre = TextExtractorPre(path / 'text', path / 'sorted_pages')
        xml_path = path / 'metadata.xml'

        number = 0
        for event, elem in etree.iterparse(xml_path, events=('end', 'start-ns')):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    number += 1
                    result = None
                    try:
                        xmlstr = etree.tostring(elem, encoding='unicode', method='xml')
                        elem.clear()
                        result = xmltodict.parse(xmlstr)
                        result = result['record']  # odstranenie record tagu
                        DataImporter.move_tag_names(result)
                        new_dict = DataImporter.extract_metadata(result)
                        if new_dict is None:
                            continue
                        if ElasticHandler.get_document(index, new_dict['id_001']) is not None:
                            continue

                        oai = new_dict.get('oai', None)
                        if oai is not None:
                            uuids = pairs.get(oai, None)
                            if uuids is not None:
                                uuid = ""
                                pre_text = ""
                                for ui in uuids:
                                    pre_text = te_pre.get_text(ui)
                                    if pre_text != "":
                                        uuid = ui
                                        break
                                if uuid != "":
                                    new_dict['uuid'] = uuid
                                if pre_text != "":
                                    new_dict['text'] = pre_text
                                    pre_text_split = pre_text.split(' ')
                                    new_dict['text_length'] = len(pre_text_split)

                    except Exception as error:
                        id_marc = ""
                        if result is not None:
                            field_001 = result.get('001', None)
                            if field_001 is not None:
                                id_marc = field_001
                        print("exception during proccesing record " + id_marc)
                        print(error)
                        new_dict = None
                        pass
                    if new_dict is not None:
                        try:
                            ElasticHandler.save_document(index, new_dict)
                        except exceptions.ElasticsearchException as elastic_error:
                            print("failed to save record: " + new_dict['id_001'])
                            print(elastic_error)
                            pass
                    if number % 10000 == 0:
                        print("processed " + str(number) + " records")
        print("import data end")

    @staticmethod
    def extract_metadata(result):
        at_least_one = ["505", "520", "521", "630"]
        if not DataImporter.is_in_language_dict(result, 'cze'):
            return None
        field_001 = result.get('001', None)
        if field_001 is None:
            return None
        field_072 = result.get('072', [])
        if isinstance(field_072, (AttrDict, dict)):
            field_072 = [field_072]
        field_080 = result.get('080', [])
        if isinstance(field_080, (AttrDict, dict)):
            field_080 = [field_080]
        field_650 = result.get('650', [])
        if isinstance(field_650, (AttrDict, dict)):
            field_650 = [field_650]
        field_oai = result.get('OAI', None)
        oai = None
        if field_oai is not None:
            oai = field_oai.get('a', None)
        field_245 = result.get('245', None)
        additional_info = ""
        for key in at_least_one:
            value = result.get(key, None)
            if value is None:
                continue
            else:
                if isinstance(value, list):
                    for a in value:
                        additional_info = additional_info + " " + str(a.get('a', ""))
                else:
                    additional_info = additional_info + " " + value.get('a', "")
        if additional_info == "":
            additional_info = None
        field_020 = result.get('020', "")
        if field_020 != "":
            if isinstance(field_020, list):
                field_020 = field_020[0].get('a', '')
            else:
                field_020 = field_020.get('a', '')
        konspekts = []
        for field in field_072:
            if field.get('2', "") == 'Konspekt':
                try:
                    konspekts.append({'category': int(field['9']), 'group': field['a']})
                except KeyError as ke:
                    continue
            else:
                continue
        if not konspekts:
            konspekts = None
        keywords = []
        for field in field_650:
            if field.get('2', '') == 'czenas':
                try:
                    if field['a'] not in keywords:
                        keywords.append(field['a'])
                except KeyError as ke:
                    continue
            else:
                continue
        if not keywords:
            keywords = None
        mdts = []
        for field in field_080:
            try:
                mdts.append(field['a'])
            except KeyError as ke:
                # print(ke)
                continue
        if not mdts:
            mdts = None
        new_dict = {'id_001': field_001, 'isbn': field_020,
                    'keywords': keywords,
                    'konspekt': konspekts, 'mdt': mdts,
                    'title': field_245.get('a', "") + " " + field_245.get('b', ""),
                    'additional_info': additional_info}
        if oai is not None:
            new_dict['oai'] = oai
        return new_dict

    def import_fulltext(self, index, to_index):
        pairs = Helper.get_pairs('data/sloucena_id')
        te = TextExtractor('data/all', 'data/sorted_pages')
        te_pre = TextExtractorPre('data/all', 'data/sorted_pages')
        client = Elasticsearch()
        s = Search(using=client, index=index)
        s.execute()
        i = 0
        for hit in s.scan():
            hit = hit.to_dict()
            try:
                field_001 = hit['001']
                exists_query = Search(using=client, index=to_index).query(Q({"match": {"id_mzk": field_001}}))
                response = exists_query.execute()
                if len(response.hits) != 0:
                    continue
                field_072 = hit['072']
                if isinstance(field_072, (AttrDict, dict)):
                    field_072 = [field_072]
                field_080 = hit['080']
                if isinstance(field_080, (AttrDict, dict)):
                    field_080 = [field_080]
                field_650 = hit['650']
                if isinstance(field_650, (AttrDict, dict)):
                    field_650 = [field_650]
                oai = hit['OAI']['a']
                field_245 = hit['245']
            except KeyError as ke:
                # print(ke)
                continue
            field_020 = hit.get('020', "")
            if field_020 != "":
                if isinstance(field_020, list):
                    field_020 = field_020[0].get('a', '')
                else:
                    field_020 = field_020.get('a', '')
            konspekts = []
            for field in field_072:
                if field.get('2', "") == 'Konspekt':
                    try:
                        konspekts.append({'category': int(field['9']), 'group': field['a']})
                    except KeyError as ke:
                        # print(ke)
                        continue
                else:
                    continue
            if len(konspekts) == 0:
                continue
            keywords = []
            for field in field_650:
                if field.get('2', '') == 'czenas':
                    try:
                        if field['a'] not in keywords:
                            keywords.append(field['a'])
                    except KeyError as ke:
                        # print(ke)
                        continue
                else:
                    continue
            if len(keywords) == 0:
                continue
            mdts = []
            for field in field_080:
                try:
                    mdts.append(field['a'])
                except KeyError as ke:
                    # print(ke)
                    continue
            uuids = pairs.get(oai, None)
            if uuids is None:
                continue
            text = ""
            uuid = ""
            pre_text = ""
            # for ui in uuids:
            #     text = te.get_text(ui)
            #     if text != "":
            #         uuid = ui
            #         break
            # if text == "":
            #     continue
            for ui in uuids:
                pre_text = te_pre.get_text(ui)
                if pre_text != "":
                    uuid = ui
                    break
            if pre_text == "":
                continue
            pre_text_split = pre_text.split(' ')
            new_dict = {'id_mzk': field_001, 'uuid': uuid, 'oai': oai, 'isbn': field_020, 'text': "",
                        'czech': pre_text,
                        'text_pre': "", 'tags_pre': [], 'tags_elastic': [], 'keywords': keywords,
                        'konpsket': konspekts, 'mdt': mdts,
                        'title': field_245.get('a', "") + " " + field_245.get('b', ""),
                        'czech_length': len(pre_text_split)}
            print(new_dict)
            print('saving number:' + str(i))
            try:
                client.index(index=to_index, body=new_dict, request_timeout=60)
            except exceptions.ElasticsearchException as elasticerror:
                print("couldnt save document uuid: " + uuid + " length of document: " + str(len(pre_text_split)))
                print(elasticerror)
                pass

            # client.indices.refresh(index=index_to)
            i += 1
            # break

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

    # selektuje data ktore obsahuju 072 + 650 a fulltext potrebne pre trenovanie
    @staticmethod
    def select_data(path, path_to):
        print("import data start")
        path = Path(path)
        path_to = Path(path_to)
        pairs = Helper.get_pairs(path / 'sloucena_id')
        te = TextExtractor(path / 'text', path / 'sorted_pages')
        xml_path = path / 'metadata.xml'

        new_file = io.open(path_to / 'new.xml', "wb+")
        xml_head = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        new_file.write(xml_head.encode("utf8"))
        new_file.write("<all>\n".encode("utf8"))

        number = 0
        for event, elem in etree.iterparse(xml_path, events=('end', 'start-ns')):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    number += 1
                    result = None
                    try:
                        xmlstr = etree.tostring(elem, encoding='unicode', method='xml')
                        result = xmltodict.parse(xmlstr)
                        result = result['record']  # odstranenie record tagu
                        DataImporter.move_tag_names(result)
                        new_dict = DataImporter.extract_metadata(result)
                        if new_dict is None:
                            elem.clear()
                            continue
                        if new_dict.get("keywords", None) is None or new_dict.get("konspekt", None) is None:
                            elem.clear()
                            continue

                        oai = new_dict.get('oai', None)
                        if oai is not None:
                            uuids = pairs.get(oai, None)
                            if uuids is not None:
                                uuid = ""
                                pre_text = ""
                                for ui in uuids:
                                    pre_text = te.get_text(ui)
                                    if pre_text != "":
                                        uuid = ui
                                        break
                                if uuid != "":
                                    print("found: " + uuid)
                                    file = uuid
                                    if "uuid" not in file:
                                        file = "uuid_" + file
                                    if ".tar.gz" not in file:
                                        file = file + ".tar.gz"
                                    file.replace(':', '_')
                                    os.system('copy ' + str(path) + '\\text\\' + file + ' ' + str(path_to) + '\\text')
                                    file = uuid
                                    if "uuid" not in file:
                                        file = "uuid_" + file
                                    if ".txt" not in file:
                                        file = file + ".txt"
                                    file.replace(':', '_')
                                    os.system('copy ' + str(path) + '\\sorted_pages\\' + file + ' ' + str(path_to) +
                                              '\\sorted_pages')
                                    new_file.write(xmlstr.encode("utf8"))

                    except Exception as error:
                        id_marc = ""
                        if result is not None:
                            field_001 = result.get('001', None)
                            if field_001 is not None:
                                id_marc = field_001
                        print("exception during proccesing record " + id_marc)
                        print(error)
                        new_dict = None
                        elem.clear()
                        pass
                    elem.clear()
                    if number % 10000 == 0:
                        print("processed " + str(number) + " records")
        new_file.write("</all>".encode("utf8"))
        new_file.close()
        print("import data end")
