import os
import tarfile
from nltk.tokenize import sent_tokenize
from preprocessor import Preprocessor
from gensim.models.doc2vec import TaggedDocument


class TextExtractor:
    """
    trieda pre extrakciu textov ktore su zabalene do .tar.gz suborov
    poskytuje iterator pre prechadzanie titulov
    """
    def __init__(self, directory, sorted_pages):
        """

        :param directory: adresar s textami v .tar.gz suboroch
        :param sorted_pages: adresar s .txt subormi ktore obsahuju poradie stran pre texty
        """
        if not os.path.isdir(directory):
            raise Exception("Not a directory: " + directory)
        if not os.path.isdir(sorted_pages):
            raise Exception("Not a directory: " + sorted_pages)
        self.directory = directory
        self.sorted_pages_path = sorted_pages
        self.sorted_pages = {}
        for file in os.listdir(self.sorted_pages_path):
            if file.endswith(".txt"):
                self.sorted_pages[file.split('.')[0]] = os.path.join(sorted_pages, file)
        self.all_files = []
        for file in os.listdir(self.directory):
            if file.endswith("tar.gz"):
                self.all_files.append(file)

    def __iter__(self):
        self.files = self.all_files.copy()
        return self

    def __next__(self):
        if len(self.files) == 0:
            raise StopIteration
        file = self.files[0]
        gz = tarfile.open(os.path.join(self.directory, file), "r:gz")
        uuid = file.split('.')[0].replace(':', '_')
        sorted_pages_path = self.sorted_pages.get(uuid, None)
        if sorted_pages_path is None:
            return ""
        sorted_pages = self.get_sorted_pages(sorted_pages_path)
        pages = {}
        for gz_member in gz.getmembers():
            if ".txt" not in gz_member.name:
                continue
            page = gz.extractfile(gz_member).read().decode(encoding="UTF-8")
            pages[gz_member.name.split('/')[1].replace('.txt', '')] = page

        text = ""
        for sorted_page in sorted_pages:
            page = pages.get(sorted_page, "").replace('\n', ' ')
            text = text + page

        gz.close()
        self.files.pop(0)
        return text

    def get_sorted_pages(self, path):
        """

        :param path: cesta k suboru so zoradenimi strankami
        :return: zoznam uuid stranok
        """
        sorted_pages = []
        with open(path, 'r') as file:
            for line in file.readlines():
                sorted_pages.append(line.replace('\n', ''))
        return sorted_pages

    def get_text(self, uuid):
        """

        :param uuid: uuid pozadovaneho textu
        :return: text pre uuid
        """
        file = uuid
        if "uuid" not in file:
            file = "uuid_" + file
        if ".tar.gz" not in file:
            file = file + ".tar.gz"
        file.replace(':', '_')
        if file not in self.all_files:
            return ""
        gz = tarfile.open(os.path.join(self.directory, file), "r:gz")
        uuid = file.split('.')[0].replace(':', '_')
        sorted_pages = self.get_sorted_pages(self.sorted_pages[uuid])
        pages = {}
        for gz_member in gz.getmembers():
            if ".txt" not in gz_member.name:
                continue
            page = gz.extractfile(gz_member).read().decode(encoding="UTF-8")
            pages[gz_member.name.split('/')[1].replace('.txt', '')] = page

        text = ""
        for sorted_page in sorted_pages:
            page = pages.get(sorted_page, "").replace('\n', ' ')
            text = text + page

        gz.close()
        return text


class TextExtractorPre:
    """
    trieda ktora je dekoratorom extraktoru textov
    spravi rozdelenie na vety a lematizaciu
    """
    def __init__(self, directory, sorted_pages, preprocess=True, uuids=None):
        self.directory = directory
        self.sorted_pages = sorted_pages
        self.processor = Preprocessor()
        self.preprocess = preprocess
        self.uuids = uuids

    def __iter__(self):
        if self.uuids is None:
            self.extractor = iter(TextExtractor(self.directory, self.sorted_pages))
        else:
            self.extractor = TextExtractor(self.directory, self.sorted_pages)
            self.uuid_iter = iter(self.uuids)
        return self

    def __next__(self):
        if self.uuids is None:
            document = next(self.extractor)
        else:
            uuid = next(self.uuid_iter)
            document = self.extractor.get_text(uuid)

        if document == "":
            return next(self)
        if self.preprocess:
            document = self.processor.remove_stop_words(document)
            document = self.processor.lemmatize(document)
        else:
            document = self.processor.tokenize(document)
        return ' '.join(document)


class TextExtractorPreTag:
    def __init__(self, directory, sorted_pages, preprocess=True):
        self.directory = directory
        self.sorted_pages = sorted_pages
        self.processor = Preprocessor()
        self.preprocess = preprocess

    def __iter__(self):
        self.extractor = iter(TextExtractor(self.directory, self.sorted_pages))
        self.index = -1
        return self

    def __next__(self):
        document = next(self.extractor)
        if document == "":
            self.index += +1
            return TaggedDocument(document, [self.index])
        if self.preprocess:
            document = self.processor.remove_stop_words(document)
            document = self.processor.lemmatize(document)
        else:
            document = self.processor.tokenize(document)
        self.index += 1
        return TaggedDocument(document, [self.index])
