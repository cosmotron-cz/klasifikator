from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
import pandas as pd
from pandas import DataFrame


def exists_at_least_one(hit_dict):
    for key in at_least_one:
        if hit_dict.get(key, None) is not None:
            return True
    return False


def transform_dict(hit_dict):
    new_dict = {}
    new_dict['001'] = hit_dict['001']
    new_dict['OAI'] = hit_dict['OAI']['a']
    if isinstance(hit_dict['072'], list):
        i = 0
        for field_072 in hit_dict['072']:
            new_dict['konspekt' + str(i)] = field_072['9']
    else:
        new_dict['konspekt0'] = hit_dict['072']['9']
    for key in at_least_one:
        value = hit_dict.get(key, None)
        if value is None:
            new_dict[key] = ""
        else:
            new_value = ""
            if isinstance(value, list):
                for a in value:
                    new_value = new_value + " " + a['a']
            else:
                new_value = value['a']
            new_dict[key] = new_value
    df = DataFrame(new_dict, index=[hit_dict['001']])
    return df


index = "records_mzk_filtered"
at_least_one = ["600", "610", "611", "630", "650", "500", "501", "502", "504", "505", "506", "507", "508", "510", "511",
                "513", "514", "515", "516", "518", "520", "521", "522", "524", "525", "526", "530", "532", "533", "534",
                "535", "536", "538", "540", "541", "542", "544", "545", "546", "547", "550", "552", "555", "556", "561",
                "562", "563", "565", "567", "580", "581", "583", "584", "585", "586", "588", "590", "595"]
es = Elasticsearch()
s = Search(using=es, index=index)
# s.from_dict({"query": {
#     "bool": {
#         "must": [],
#         "filter": [
#             {
#                 "bool": {
#                     "should": [
#                         {
#                             "bool": {
#                                 "should": [
#                                     {
#                                         "exists": {
#                                             "field": "600.a"
#                                         }
#                                     }
#                                 ],
#                                 "minimum_should_match": 1
#                             }
#                         },
#                         {
#                             "bool": {
#                                 "should": [
#                                     {
#                                         "bool": {
#                                             "should": [
#                                                 {
#                                                     "exists": {
#                                                         "field": "610.a"
#                                                     }
#                                                 }
#                                             ],
#                                             "minimum_should_match": 1
#                                         }
#                                     },
#                                     {
#                                         "bool": {
#                                             "should": [
#                                                 {
#                                                     "bool": {
#                                                         "should": [
#                                                             {
#                                                                 "exists": {
#                                                                     "field": "611.a"
#                                                                 }
#                                                             }
#                                                         ],
#                                                         "minimum_should_match": 1
#                                                     }
#                                                 },
#                                                 {
#                                                     "bool": {
#                                                         "should": [
#                                                             {
#                                                                 "bool": {
#                                                                     "should": [
#                                                                         {
#                                                                             "exists": {
#                                                                                 "field": "650.a"
#                                                                             }
#                                                                         }
#                                                                     ],
#                                                                     "minimum_should_match": 1
#                                                                 }
#                                                             },
#                                                             {
#                                                                 "bool": {
#                                                                     "should": [
#                                                                         {
#                                                                             "exists": {
#                                                                                 "field": "630.a"
#                                                                             }
#                                                                         }
#                                                                     ],
#                                                                     "minimum_should_match": 1
#                                                                 }
#                                                             }
#                                                         ],
#                                                         "minimum_should_match": 1
#                                                     }
#                                                 }
#                                             ],
#                                             "minimum_should_match": 1
#                                         }
#                                     }
#                                 ],
#                                 "minimum_should_match": 1
#                             }
#                         }
#                     ],
#                     "minimum_should_match": 1
#                 }
#             }
#         ],
#         "should": [],
#         "must_not": []
#     }
# }})
s = s.source(["001", "OAI", "072"] + at_least_one)

s.execute()
dataframes = []
for hit in s:
    hit_dict = hit.to_dict()
    if exists_at_least_one(hit_dict):
        df = transform_dict(hit_dict)
        dataframes.append(df)
data = pd.concat(dataframes)
print(data)
data.to_csv(r'C:\Users\jakub\Documents\medzivysledok.csv')
