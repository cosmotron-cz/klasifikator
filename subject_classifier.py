from classifier_keywords import ClassifierKeywords
from classifier_fulltext import ClassifierFulltext
from match_konspect import MatchKonspekt
from elastic_handler import ElasticHandler
from preprocessor import Preprocessor
from keywords_generator import KeywordsGenerator


class SubjectClassifier:
    def __init__(self):
        self.classifier_fulltext = ClassifierFulltext()
        self.classifier_keywords = ClassifierKeywords()
        self.match_konspekt = MatchKonspekt()
        self.elastic_handler = ElasticHandler()
        self.preprocessor = Preprocessor()
        self.keyword_generator = KeywordsGenerator()

    def classify_document(self, metadata, fulltext):
        # prepossessing fulltext and save document to elastic
        fulltext_pre = self.preprocessor.remove_stop_words(fulltext)
        fulltext_pre = self.preprocessor.lemmatize(fulltext_pre)
        id_elastic = self.elastic_handler.save_document(metadata, fulltext, fulltext_pre)
        if id_elastic is None:
            raise Exception('Could not save document')

        konspekt = None
        keywords = None
        if metadata['mdt'] is not None:
            # generate konspekt from mdt
            konspekt = self.match_konspekt.find_and_choose(metadata['mdt'])  # TODO skontolovat

        if konspekt is None and metadata.get('keywords', None) is not None:
            # generate konspekt from keywords
            konspekt = self.classifier_keywords.classify_elastic(id_elastic)  # TODO skontolovat

        if konspekt is None:
            # generate konspekt from fulltext
            konspekt = self.classifier_fulltext.classify_elastic(id_elastic)  # TODO skontolovat

        if metadata.get('keywords', None) is None:
            # generate keywords from fulltext
            keywords = self.keyword_generator.generate_kewords_elastic(id_elastic)  # TODO skontolovat

        return konspekt, keywords
