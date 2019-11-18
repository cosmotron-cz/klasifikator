import os
import re

class MatchKonspekt():
    def __init__(self):
        self.rules = {}
        k = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "/rules.txt", "r", encoding="utf8") as file:
            slash_mdt = []
            for line in file:
                parts = line.split("$$")
                if "b1" == parts[2]:
                    k += 1
                    continue

                pattern = parts[1][1:]
                desc = parts[3][1:]
                new_dict = {"category": k, "description": desc, "original": ""}
                self.rules[pattern] = new_dict
                if pattern.find('/') != -1:
                    slash_mdt.append(pattern)
        for pattern in slash_mdt:
            unpacked = self.unpack_mdt(pattern)
            original = self.rules[pattern]
            if len(unpacked) == 1:
                continue
            for mdt in unpacked:
                if self.rules.get(mdt, None) is not None:
                    continue
                new_unpacked_dict = {"category": original['category'], "description": original['description'],
                                     "original": pattern}
                self.rules[mdt] = new_unpacked_dict

    def find_and_choose(self, mdts):
        result = [None, None, None]
        all_konspekts = []
        second = ["37.016", "01", "929", "8-93", "821-93", "09", "912"]
        for mdt in mdts:
            category, subcategory, description = self.find_category(mdt)
            if category != -1:
                if result[0] is None:
                    result[0] = {"category": category, "subcategory": subcategory, "description": description}
                    continue
                all_konspekts.append({"category": category, "subcategory": subcategory, "description": description})
        for found in all_konspekts:
            if result[2] is None and found['subcategory'] != result[0]['subcategory'] and \
                    self.is_childs_literature(found["subcategory"]):
                rule = self.rules.get(found["subcategory"], None)
                result[2] = {"category": rule["category"], "subcategory": found["subcategory"],
                             "description": rule["description"]}
                continue
            if result[2] is None and found['subcategory'] != result[0]['subcategory'] and \
                    found["subcategory"] in second:
                rule = self.rules.get(found["subcategory"], None)
                result[2] = {"category": rule["category"], "subcategory": found["subcategory"],
                             "description": rule["description"]}
                continue
            if result[1] is None and found['category'] != result[0]['category'] and \
                    (result[2] is None or found['subcategory'] != result[2]['subcategory']):
                rule = self.rules.get(found["subcategory"], None)
                result[1] = {"category": rule["category"], "subcategory": found["subcategory"],
                             "description": rule["description"]}
        return result[0], result[1], result[2]

    def find_category(self, mdt):
        mdt = re.sub('[^\d.+\/:\[\]=()\"-]+', '', mdt)
        found_slash = re.search(r"^[\.\d]+/[\.\-\d]\d+", mdt)
        if found_slash is not None:
            slash_mdt = mdt
            right = found_slash.end()
            while len(slash_mdt) >= right:
                match = self.rules.get(slash_mdt, None)
                if match is not None:
                    if match["original"] != "":
                        return match["category"], match["original"], match["description"]
                    else:
                        return match["category"], slash_mdt, match["description"]
                slash_mdt = self.shorten_from_right(slash_mdt)
        unpacked = self.unpack_mdt(mdt)
        for u in unpacked:
            while u != "":
                match = self.rules.get(u, None)
                if match is not None:
                    if match["original"] != "":
                        return match["category"], match["original"], match["description"]
                    else:
                        return match["category"], u, match["description"]
                u = self.shorten_from_right(u)
        return -1, "", ""

    def compare_mdt(self, mdt1, mdt2):
        mdt1_cat_len = self.length_category(mdt1)
        mdt2_cat_len = self.length_category(mdt2)

        if mdt1_cat_len > mdt2_cat_len:
            return mdt1
        elif mdt2_cat_len > mdt1_cat_len:
            return mdt2
        else:
            if len(mdt1) > mdt1_cat_len and mdt1[mdt1_cat_len] == '/':
                return mdt2
            elif len(mdt2) > mdt2_cat_len and mdt2[mdt1_cat_len] == '/':
                return mdt1
            else:
                if len(mdt1) >= len(mdt2):
                    return mdt1
                else:
                    return mdt2

    def length_category(self, mdt):
        mdt_cat_len = 0
        for c in mdt:
            if c.isdigit():
                mdt_cat_len += 1
            elif c == '.' and mdt_cat_len % 4 == 3:
                mdt_cat_len += 1
            else:
                break
        return mdt_cat_len

    def unpack_mdt(self, mdt):
        unpacked = []
        if mdt.find('+') != -1:
            splitted_plus = mdt.split('+')
            for splitted in splitted_plus:
                unpacked_plus = self.unpack_mdt(splitted)
                unpacked = unpacked + unpacked_plus
                return unpacked

        found_numbers_only = re.search(r"\d+/\d+", mdt)
        found_with_dot = re.search(r"\.\d+/\.\d+", mdt)
        sign = ''
        if found_numbers_only is not None:
            left, right = found_numbers_only.span()
            if (left > 0 and mdt[left-1] == '"') or (right < len(mdt) - 1 and mdt[right] == '"'):
                return [mdt]
            bot, top = mdt[left:right].split('/')

        elif found_with_dot is not None:
            left, right = found_with_dot.span()
            if (left > 0 and mdt[left - 1] == '"') or (right < len(mdt) - 1 and mdt[right] == '"'):
                return [mdt]
            part = mdt[left + 1:right]
            bot, top = part.split('/.')
            sign = '.'

        else:
            return [mdt]

        if len(bot) != len(top):
            return [mdt]
        number_length = len(bot)
        try:
            bot = int(bot)
            top = int(top)
        except ValueError as ve:
            print(ve)
            print("chyba: " + mdt)
            return [mdt]

        for i in range(bot, top+1):
            str_i = self.fill_zeros(i, number_length)
            new_mdt = mdt[:left] + sign + str_i + mdt[right:]
            unpacked.append(new_mdt)
            if new_mdt.find('/') != -1:
                children = self.unpack_mdt(new_mdt)
                unpacked = unpacked + children

        return unpacked

    def fill_zeros(self, number, length):
        str_number = str(number)
        if len(str_number) == length:
            return str_number
        times = length - len(str_number)
        return times * '0' + str_number

    def shorten_from_right(self, mdt):
        length = len(mdt)
        if mdt[length-1] == '"':
            position = mdt.find('"')
            return mdt[:position]
        elif mdt[length-1] == ')':
            position = mdt.find('(')
            if mdt.count('(') > 1:
                return mdt[:position]
            if position == length - 2 or position == length - 3:
                return mdt[:position]
            else:
                return mdt[:length-2] + mdt[length-1:]
        elif mdt[length-1] == ']':
            position = mdt.find('[')
            return mdt[:position]
        elif mdt[length-1].isdigit():
            if mdt[length-2].isdigit():
                return mdt[:length-1]
            else:
                if mdt[length-3] == '/':
                    return mdt[:length-3]
                else:
                    return mdt[:length-2]
        else:
            return mdt[:length - 1]

    def is_childs_literature(self, mdt):
        if "-053.2" in mdt or "(0.053.2)" in mdt:
            return True
        else:
            return False
