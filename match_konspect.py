import os
import re

class MatchKonspekt():
    def __init__(self):
        self.rules = {}
        k = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "\\rules.txt", "r", encoding="utf8") as file:
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
                    unpacked = self.unpack_mdt(pattern)
                    if len(unpacked) == 1:
                        continue
                    for mdt in unpacked:
                        new_unpacked_dict = {"category": k, "description": desc, "original": pattern}
                        self.rules[mdt] = new_unpacked_dict

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
            if (left > 0 and (mdt[left-1] == '(' or mdt[left-1] == '"')) or \
                    (right < len(mdt) - 1 and (mdt[right] == ')' or mdt[right] == '"')):
                return [mdt]
            bot, top = mdt[left:right].split('/')

        elif found_with_dot is not None:
            left, right = found_with_dot.span()
            if (left > 0 and (mdt[left - 1] == '(' or mdt[left - 1] == '"')) or \
                    (right < len(mdt) - 1 and (mdt[right] == ')' or mdt[right] == '"')):
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

    def unpack_mdt2(self, mdt):
        position = mdt.find('/')
        sign = ""
        if position == -1:
            return []
        if mdt[position+1] != '.' and mdt[position+1] != '-':
            found_parenthesis = re.search(r"\(.+/.+\)", mdt)
            found_quotes = re.search(r"\".+/.+\"", mdt)
            if found_parenthesis is not None:
                left, right = found_parenthesis.span()
                right -= 1
                bot, top = mdt[left+1:right].split('/')
            elif found_quotes is not None:
                left, right = found_quotes.span()
                right -= 1
                bot, top = mdt[left + 1:right].split('/')
            else:
                bot, top = mdt.split('/')
                left = 0
                right = len(mdt)
        elif mdt[position+2].isdigit():
            sign = mdt[position + 1]
            left = position - 1
            right = position + 2
            while mdt[left] != sign:
                left -= 1
            while right < len(mdt) and mdt[right].isdigit():
                right += 1
            part = mdt[left+1:right]
            bot, top = part.split('/' + sign)
        else:
            print("neviem co robit")
            return [mdt]
        if len(bot) != len(top):
            print("dorobit :" + mdt)
            return []
            # raise Exception("horna a dolna hranica nemaju rovnaku dlzku: " + mdt)
        number_lenght = len(bot)
        try:
            bot = int(bot)
            top = int(top)
        except ValueError as ve:
            print(ve)
            print("chyba: " + mdt)
            return []
        unpacked = []
        for i in range(bot, top+1):
            if mdt[left] == '(' or mdt[left] == '"':
                str_i = self.fill_zeros(i, number_lenght)
                new_mdt = mdt[:left+1] + sign + str_i + mdt[right:]
            else:
                str_i = self.fill_zeros(i, number_lenght)
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
            return mdt[:position]
        elif mdt[length-1] == ']':
            position = mdt.find('[')
            return mdt[:position]
        elif mdt[length - 1] == ':':
            return mdt[:length-1]
        elif mdt[length - 1] == '(':
            return mdt[:length-1]
        elif mdt[length - 1] == '[':
            return mdt[:length-1]
        elif mdt[length - 1] == '.':
            return mdt[:length-1]
        elif mdt[length - 1] == '\'':
            return mdt[:length-1]
        elif mdt[length - 1] == '/':
            return mdt[:length-1]
        elif mdt[length - 1] == '-':
            return mdt[:length-1]
        elif mdt[length - 1] == '=':
            return mdt[:length - 1]
        elif mdt[length-1].isdigit():
            if mdt[length-2].isdigit():
                return mdt[:length-1]
            else:
                if mdt[length-3] == '/':
                    return mdt[:length-3]
                else:
                    return mdt[:length-2]
        elif mdt.find(' ') != -1:
            splitted = mdt.split(' ')
            return splitted[0]
        else:
            raise Exception("chyba pri spracovani mdt - na konci necakany znak: " + mdt)

    def find_category(self, mdt):
        mdt = re.sub('[^\d.+\/:\[\]=()\"-]+', '', mdt)
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
