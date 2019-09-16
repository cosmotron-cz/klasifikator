import re
from lxml import etree
import io
from elasticsearch import Elasticsearch
from elasticsearch import exceptions
from match_konspect import MatchKonspekt

class DataExporter:

    # opravy poradie atributov v datafieldoch
    # NOT FINISHED
    # MIGHT DELETE
    @staticmethod
    def change_order_attr(path):
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                if line.find("<datafield") != -1:
                    tag = re.search(r"tag=\".{3}\"", line)
                    ind1 = re.search(r"ind1=\".\"", line)
                    ind2 = re.search(r"ind2=\".\"", line)
                    if tag is None:
                        print("nenasiel sa tag")
                        break
                    beginning_line = re.search(r"^.*<datafield", line)
                    new_line = beginning_line.group(0) + " " + tag.group(0)
                    if ind1 is not None:
                        new_line += " " + ind1.group(0)
                    if ind2 is not None:
                        new_line += " " + ind2.group(0)
                    print(new_line)
                else:
                    print(line)

    # prida konspekt do xml, uklada do noveho suboru
    @staticmethod
    def add_konspect_xml(path_from, path_to):
        ns = "{http://www.loc.gov/MARC21/slim}"
        new_file = io.open(path_to, "wb+")
        mk = MatchKonspekt()
        xml_head = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        new_file.write(xml_head.encode("utf8"))
        new_file.write("<all>\n".encode("utf8"))
        for event, elem in etree.iterparse(path_from, events=('end', 'start-ns'), remove_blank_text=True):
            if event == 'end':
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]  # odstranenie namespace
                if elem.tag == "record":
                    found = elem.findall("./datafield[@tag='080']/subfield[@code='a']")
                    found_072 = elem.xpath("./datafield[@tag='072']/subfield[contains(text(),'Konspekt')]")
                    if found != [] and found_072 == []:
                        konspecs = {}
                        mdts = []
                        for field_080 in found:
                            mdts.append(field_080.text)
                        konspekt_one, konspekt_two, konspekt_three = mk.find_and_choose(mdts)
                        if konspekt_one is None and konspekt_two is None and konspekt_three is None:
                            continue
                        if konspekt_one is not None:
                            konspecs[konspekt_one['category']] = {'subcategory': konspekt_one['subcategory'],
                                                                        "description": konspekt_one['description']}
                        if konspekt_two is not None:
                            konspecs[konspekt_two['category']] = {'subcategory': konspekt_two['subcategory'],
                                                                  "description": konspekt_two['description']}
                        if konspekt_three is not None:
                            konspecs[konspekt_three['category']] = {'subcategory': konspekt_three['subcategory'],
                                                                  "description": konspekt_three['description']}
                        for key, val in konspecs.items():
                            new_072 = DataExporter.create_072(key, val['subcategory'], val['description'])
                            elem.append(new_072)
                        if len(konspecs) > 0:
                            new_file.write(etree.tostring(elem, encoding="utf8", xml_declaration=False, method="xml", pretty_print=True))
                    elem.clear()
        new_file.write("</all>".encode("utf8"))
        new_file.close()

    @staticmethod
    def create_072(category, subcategory, description):
        field_072 = etree.Element("datafield")
        field_072.set("tag", "072")
        field_072.set("ind1", " ")
        field_072.set("ind2", "7")

        subfield_a = etree.Element("subfield")
        subfield_a.set("code", 'a')
        subfield_a.text = subcategory
        field_072.append(subfield_a)

        subfield_x = etree.Element("subfield")
        subfield_x.set("code", 'x')
        subfield_x.text = description
        field_072.append(subfield_x)

        subfield_2 = etree.Element("subfield")
        subfield_2.set("code", '2')
        subfield_2.text = "Konspekt_auto"
        field_072.append(subfield_2)

        subfield_9 = etree.Element("subfield")
        subfield_9.set("code", '9')
        subfield_9.text = str(category)
        field_072.append(subfield_9)

        return field_072

path = 'C:\\Users\\jakub\\Documents\\metadata_mzk.xml'
path_to = 'C:\\Users\\jakub\\Documents\\export4_mzk.xml'
DataExporter.add_konspect_xml(path, path_to)
#DataExporter.change_order_attr(path)