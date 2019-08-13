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
        three = 0
        for hit in s.scan():
            found_konspects = []
            field_072 = hit['072']
            if isinstance(field_072, AttrDict):
                field_072 = [field_072]
            field_080 = hit['080']
            if isinstance(field_080, AttrDict):
                field_080 = [field_080]
            mdts = []
            for mdt in field_080:
                try:
                    mdts.append(mdt['a'])
                except KeyError as ke:
                    continue
            try:
                konspekt_one, konspekt_two, konspekt_three = mk.find_and_choose(mdts)
            except Exception as e:
                print(e)
                continue
            if konspekt_one is not None:
                found_konspects.append(konspekt_one)
            if konspekt_two is not None:
                found_konspects.append(konspekt_two)
            if konspekt_three is not None:
                found_konspects.append(konspekt_three)
            if konspekt_one is None and konspekt_two is None and konspekt_three is None:
                continue
            all_konspect += 1
            already_tested = []
            for real in field_072:
                try:
                    if real['2'] != "Konspekt":
                        continue
                    if real['9'] == "" or real['a'] == "":
                        continue
                except KeyError as ke:
                    #print("didnt pass control, missing field: " + str(ke))
                    continue
                all_category += 1
                all_subcategory += 1
                for konspect in found_konspects:
                    try:
                        if konspect['category'] not in already_tested:
                            if str(konspect['category']) == real['9']:
                                correct_category += 1
                                already_tested.append(konspect['category'])
                                if konspect['subcategory'] == real['a']:
                                    correct_subcategory += 1
                                # else:
                                #     print("category: " + str(konspect['category']) + " real subcategory: " + real['a'] + " auto category: " + konspect['subcategory'] + " mdt: " + str(field_080))
                    except KeyError as ke:
                        print("dict doesnt contain: " + str(ke))
                        break
            if len(already_tested) == len(field_072):
                correct_konspect += 1
            if len(field_072) > 2:
                three += 1
        print(correct_category/all_category)
        print(correct_subcategory/all_subcategory)
        print(correct_konspect/all_konspect)
        print(all_konspect)
        print(all_category)
        print(three)





t = Tester
t.test_rules("records_mzk_filtered")
