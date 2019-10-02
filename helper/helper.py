from pathlib import Path
from datetime import datetime
import os
import errno

class Helper:
    stop_words = ["a sice", "a to", "a", "aby", "aj", "ale", "ani", "aniz", "aniž", "ano", "asi", "az",
                               "až", "bez", "bude", "budem", "budes", "budeš", "by", "byl", "byla", "byli", "bylo",
                               "byt", "být", "ci", "clanek", "clanku", "clanky", "co", "coz", "což", "cz", "či",
                               "článek", "článku", "článků", "články", "dalsi", "další", "diskuse", "diskuze", "dnes",
                               "do", "fora", "fóra", "forum", "fórum", "ho", "i", "ja", "já", "jak", "jako", "je",
                               "jeho", "jej", "jeji", "její", "jejich", "jen", "jenz", "jenž", "jeste", "ještě", "ji",
                               "jí", "jine", "jiné", "jiz", "již", "jsem", "jses", "jseš", "jsme", "jsou", "jste", "k",
                               "kam", "kazdy", "každý", "kde", "kdo", "kdyz", "když", "ke", "ktera", "která", "ktere",
                               "které", "kteri", "kterou", "ktery", "který", "kteří", "ku", "ma", "má", "mate", "máte",
                               "me", "mě", "mezi", "mi", "mit", "mít", "mne", "mně", "mnou", "muj", "můj", "muze",
                               "může", "my", "na", "ná", "nad", "nam", "nám", "napiste", "napište", "nas", "nasi",
                               "náš", "naši", "ne", "nebo", "necht", "nechť", "nejsou", "neni", "není", "nez", "než",
                               "ni", "ní", "nic", "nove", "nové", "novy", "nový", "o", "od", "ode", "on", "pak", "po",
                               "pod", "podle", "pokud", "polozka", "polozky", "položka", "položky", "pouze", "prave",
                               "právě", "pred", "prede", "pres", "pri", "prispevek", "prispevku", "prispevky", "pro",
                               "proc", "proč", "proto", "protoze", "protože", "prvni", "první", "před", "přede", "přes",
                               "při", "příspěvek", "příspěvku", "příspěvků", "příspěvky", "pta", "ptá", "ptat", "ptát",
                               "re", "s", "se", "si", "sice", "strana", "sve", "své", "svuj", "svůj", "svych", "svých",
                               "svym", "svým", "svymi", "svými", "ta", "tak", "take", "také", "takze", "takže", "tato",
                               "te", "té", "tedy", "tema", "téma", "těma", "temat", "témat", "temata", "témata",
                               "tematem", "tématem", "tematu", "tématu", "tematy", "tématy", "ten", "tento", "teto",
                               "této", "tim", "tím", "timto", "tímto", "tipy", "to", "tohle", "toho", "tohoto", "tom",
                               "tomto", "tomuto", "toto", "tu", "tuto", "tvuj", "tvůj", "ty", "tyto", "u", "uz", "už",
                               "v", "vam", "vám", "vas", "vase", "váš", "vaše", "ve", "vice", "více", "vsak", "vsechen",
                               "však", "všechen", "vy", "z", "za", "zda", "zde", "ze", "ze", "zpet", "zpět", "zprav",
                               "zpráv", "zprava", "zpráva", "zpravy", "zprávy", "že"]
    @staticmethod
    def get_pairs(path):

        pairs = {}
        with open(path, 'r') as file:
            for line in file:
                line = line.rstrip()
                parts = line.split(',')
                keys = []
                values = []
                for part in parts:
                    if part.startswith('uuid'):
                        values.append(part.replace(':', '_'))
                    else:
                        keys.append(part)
                for key in keys:
                    pairs[key] = values
        return pairs

    @staticmethod
    def save_document(document, current_uuid, result_dir):
        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        dir = Path(result_dir)
        file = dir / 'processed' / current_uuid
        with open(file, 'w+', encoding="utf-8") as f:
            f.write(document)

        return file

    @staticmethod
    def check_processed(current_uuid, directory):

        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        file = directory / 'processed' / current_uuid
        if file.is_file():
            with open(file, 'r', encoding="utf-8") as f:
                document = f.read()
        else:
            return None
        return document

    @staticmethod
    def create_results_dir(path=None):
        if path is None:
            date_now = datetime.now()
            results_dir = Path(date_now.strftime('%Y_%m_%d_%H_%M'))
        elif isinstance(path, Path):
            results_dir = path
        elif isinstance(path, str):
            results_dir = Path(path)
        else:
            raise Exception("wrong path argument")
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return results_dir