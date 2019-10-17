from ufal.morphodita import *
from pathlib import Path
from helper.helper import Helper
from elasticsearch import Elasticsearch


class Preprocessor(object):
    def __init__(self, dictionary=None, stop_words=None, tagger=None):
        dir_path = Path(__file__).resolve().parent
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            self.dictionary = str(dir_path / "dict/czech-morfflex-161115-pos_only.dict")
        if stop_words is not None:
            self.stop_words = stop_words
        else:
            self.stop_words = Helper.stop_words

        # if tagger is not None:
        #     self.tagger = tagger
        # else:
        #     self.tagger = Tagger.load(str(dir_path / "dict/czech-morfflex-pdt-161115-pos_only.tagger"))

        self.morpho = Morpho.load(self.dictionary)
        if not self.morpho:
            raise Exception("Cannot load dictionary " + self.dictionary)
        self.tokenizer = self.morpho.newTokenizer()

    def lemmatize(self, text):
        if isinstance(text, str):
            text = self.tokenize(text)

        result = []
        lemmas = TaggedLemmas()
        converter = TagsetConverter.newPdtToConll2009Converter()
        for word in text:
            self.morpho.analyze(word, self.morpho.GUESSER, lemmas)
            converter.convert(lemmas[0])
            result.append(lemmas[0].lemma)
        return result

    def pos_tag(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        result = []
        forms = Forms()
        lemmas = TaggedLemmas()
        tokens = TokenRanges()
        tokenizer = self.tagger.newTokenizer()
        tokenizer.setText(text)
        while tokenizer.nextSentence(forms, tokens):
            self.tagger.tag(forms, lemmas)
            for i in range(len(lemmas)):
                lemma = lemmas[i]
                result.append(lemma.tag)
        return result

    def remove_stop_words(self, text):
        if isinstance(text, str):
            text = self.tokenize(text)

        result = []

        for word in text:
            if word not in self.stop_words and 1 < len(word) < 25:
                result.append(word)

        return result

    def tokenize(self, text):
        self.tokenizer.setText(text)
        forms = Forms()
        ranges = TokenRanges()
        tokens = []
        while self.tokenizer.nextSentence(forms, ranges):
            for word in forms:
                tokens.append(word.lower())

        return tokens

    @staticmethod
    def preprocess_text_elastic(text, index_to, analyzer):
        es = Elasticsearch()
        all_words = text.split()
        result = ""
        for n in range(0, len(all_words), 10000):
            body = {
              "analyzer": analyzer,
              "filter": ["unique_on_same_position"],
              "text": ' '.join(all_words[n: n+10000])
            }

            response = es.indices.analyze(index=index_to, body=body)
            if len(response['tokens']) == 0:
                break
            for token in response['tokens']:
                result += " " + token['token']
        return result