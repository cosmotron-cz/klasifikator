from match_konspect import MatchKonspekt
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.utils import AttrDict

class Tester:

    @staticmethod
    def test_rules(index):
        client = Elasticsearch()
        s = Search(using=client, index=index)
        s.execute()
        mk = MatchKonspekt()
        correct_category = 0
        all_category = 0
        correct_subcategory = 0
        all_subcategory = 0
        correct_konspect = 0
        all_konspect = 0
        for hit in s.scan():
            found_konspects = []
            field_072 = hit['072']
            if isinstance(field_072, AttrDict):
                field_072 = [field_072]
            field_080 = hit['080']
            if isinstance(field_080, AttrDict):
                field_080 = [field_080]
            for mdt in field_080:
                try:
                    category, subcategory, description = mk.find_category(mdt['a'])
                    if category == -1:
                        continue
                except Exception as e:
                    print(e)
                    continue
                new_found = {"category": category, "subcategory": subcategory, "description": description}
                found_konspects.append(new_found)
            all_konspect += 1
            already_tested = []
            for real in field_072:
                try:
                    if real['2'] != "Konspekt":
                        continue
                except KeyError as ke:
                    print("dict doesnt contain: " + str(ke))
                    continue
                all_category += 1
                all_subcategory += 1
                for konspect in found_konspects:
                    try:
                        if konspect['category'] not in already_tested:
                            if konspect['category'] == real['9']:
                                correct_category += 1
                                already_tested.append(konspect['category'])
                        if konspect['subcategory'] == real['a']:
                            correct_subcategory += 1
                    except KeyError as ke:
                        print("dict doesnt contain: " + str(ke))
                        break
            if len(already_tested) == len(field_072):
                correct_konspect += 1
        print(correct_category/all_category)
        print(correct_subcategory/all_subcategory)
        print(correct_konspect/all_konspect)





t = Tester
t.test_rules("records_mzk_filtered")
