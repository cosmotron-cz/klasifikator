from ufal.morphodita import *
from pathlib import Path
from helper.helper import Helper


class Preprocessor(object):
    def __init__(self, dictionary=None, stop_words=None):
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            dir_path = Path(__file__).resolve().parent
            self.dictionary = str(dir_path / "dict/czech-morfflex-161115-pos_only.dict")
        if stop_words is not None:
            self.stop_words = stop_words
        else:
            self.stop_words = Helper.stop_words

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
        if isinstance(text, str):
            text = self.tokenize(text)

        result = []
        lemmas = TaggedLemmas()
        for word in text:
            self.morpho.analyze(word, self.morpho.GUESSER, lemmas)
            result.append(lemmas[0].tag)
        if len(result) == 0:
            result.append('X@')
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



