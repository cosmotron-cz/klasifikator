import unittest
from subject_classifier import ClassifierKeywords, ClassifierFulltext
from preprocessor import Preprocessor


class TestClassifiers(unittest.TestCase):
    def test_keywords_classify(self):
        preprocessor = Preprocessor()
        classifier_keywords = ClassifierKeywords(preprocessor)
        category, group = classifier_keywords.classify(['daň z příjmů ze závislé činnosti'])
        if category is None:
            self.fail("classify didn't return category")
        try:
            int(category)
        except ValueError:
            self.fail("classify didn't return numeric category")

        if group is None:
            self.fail("classify didn't return group")

        if not isinstance(group, str):
            self.fail("classify didn't return group as string")

    def test_fulltext_classify(self):
        preprocessor = Preprocessor()
        classifier_keywords = ClassifierFulltext(preprocessor)
        term_vectors = {"artuš": {
            "doc_freq": 1,
            "ttf": 82,
            "term_freq": 82,
            "score": 262.17242
        },
            "bojovník": {
                "doc_freq": 2,
                "ttf": 33,
                "term_freq": 30,
                "score": 83.752785
            },
            "carduel": {
                "doc_freq": 1,
                "ttf": 24,
                "term_freq": 24,
                "score": 76.73339
            },
            "logrie": {
                "doc_freq": 1,
                "ttf": 73,
                "term_freq": 73,
                "score": 233.3974
            },
            "merlinout": {
                "doc_freq": 1,
                "ttf": 30,
                "term_freq": 30,
                "score": 95.91674
            },
            "morgan": {
                "doc_freq": 1,
                "ttf": 49,
                "term_freq": 49,
                "score": 156.664
            },
            "mrtvý": {
                "doc_freq": 5,
                "ttf": 57,
                "term_freq": 37,
                "score": 77.64866
            },
            "uther": {
                "doc_freq": 1,
                "ttf": 53,
                "term_freq": 53,
                "score": 169.45291
            },
            "velitel": {
                "doc_freq": 2,
                "ttf": 32,
                "term_freq": 31,
                "score": 86.54454
            },
            "všechen": {
                "doc_freq": 16,
                "ttf": 1499,
                "term_freq": 98,
                "score": 103.60153
            }}
        doc_count = 17
        word_count = 10575
        category, group = classifier_keywords.classify(term_vectors, word_count, doc_count)
        if category is None:
            self.fail("classify didn't return category")
        try:
            int(category)
        except ValueError:
            self.fail("classify didn't return numeric category")

        if group is None:
            self.fail("classify didn't return group")

        if not isinstance(group, str):
            self.fail("classify didn't return group as string")


if __name__ == '__main__':
    unittest.main()
