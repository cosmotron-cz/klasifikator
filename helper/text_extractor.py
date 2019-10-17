import os
import tarfile
from preprocessor import Preprocessor
from gensim.models.doc2vec import TaggedDocument
import errno
from pathlib import Path
from helper.helper import Helper
from multiprocessing import Process


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
            raise Exception("Not a directory: " + str(directory))
        if not os.path.isdir(sorted_pages):
            raise Exception("Not a directory: " + str(sorted_pages))
        self.directory = directory
        self.sorted_pages_path = sorted_pages
        self.sorted_pages = {}
        for file in os.listdir(self.sorted_pages_path):
            if file.endswith(".txt"):
                self.sorted_pages[file.split('.')[0]] = os.path.join(sorted_pages, file)
        self.all_files = []
        for file in os.listdir(self.directory):
            if file.endswith(".tar.gz"):
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
        text = text.replace('\n', ' ')
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
        text = text.replace('\n', ' ')
        return text


class TextExtractorPre:
    """
    trieda ktora je dekoratorom extraktoru textov
    spravi rozdelenie na vety a lematizaciu
    """
    def __init__(self, directory, sorted_pages, preprocess=True, uuids=None, filter_nouns=False):
        self.directory = Path(directory)
        self.sorted_pages = Path(sorted_pages)
        self.processor = Preprocessor()
        self.preprocess = preprocess
        self.uuids = uuids
        self.filter_nouns = filter_nouns
        self.extractor = TextExtractor(self.directory, self.sorted_pages)
        try:
            os.makedirs(self.directory / 'processed')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def __iter__(self):
        self.number = 0
        if self.uuids is None:
            self.extractor = iter(TextExtractor(self.directory, self.sorted_pages))
        else:
            self.extractor = TextExtractor(self.directory, self.sorted_pages)
            self.uuid_iter = iter(self.uuids)
        return self

    def __next__(self):
        print("proccesing number: " + str(self.number))
        current_uuid = ""
        try:
            processed = False
            if self.uuids is None:
                current_uuid = self.extractor.all_files[0]
                document = self.check_processed(current_uuid)
                if document is None:
                    document = next(self.extractor)
                else:
                    processed = True
            else:
                current_uuid = next(self.uuid_iter)
                document = self.check_processed(current_uuid)
                if document is None:
                    document = self.extractor.get_text(current_uuid)
                else:
                    processed = True
                if self.filter_nouns:
                    document = document.split(' ')
                    document = Helper.filter_words(document, self.processor)
                    document = ' '.join(document)

            if document == "":
                return ""
            if processed is False:
                if self.preprocess:
                    document = self.processor.remove_stop_words(document)
                    document = self.processor.lemmatize(document)
                    if self.filter_nouns:
                        document = Helper.filter_words(document, self.processor)
                else:
                    document = self.processor.tokenize(document)
                document = ' '.join(document)
                self.save_document(document, current_uuid)
            self.number += 1
        except KeyError as err:
            print(err)
            document = ""
            pass
        return document

    def get_text(self, uuid):
        print("proccesing file: " + uuid)
        try:
            processed = False
            document = self.check_processed(uuid)
            if document is None:
                document = self.extractor.get_text(uuid)
            else:
                processed = True
            if processed is False:
                if self.preprocess:
                    document = self.processor.remove_stop_words(document)
                    document = self.processor.lemmatize(document)
                    if self.filter_nouns:
                        document = Helper.filter_words(document, self.processor)
                else:
                    document = self.processor.tokenize(document)
                document = ' '.join(document)
                self.save_document(document, uuid)
        except KeyError as err:
            print(err)
            document = ""
            pass
        return "d"

    def check_processed(self, current_uuid):

        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        file = self.directory / 'processed' / current_uuid
        if file.is_file():
            with open(file, 'r', encoding="utf-8") as f:
                document = f.read()
        else:
            return None
        document = self.processor.tokenize(document)
        result = []
        for word in document:
            if len(word) > 4:
                result.append(word)
        document = ' '.join(result)
        return document

    def save_document(self, document, current_uuid):
        if current_uuid.endswith('.tar.gz'):
            current_uuid = current_uuid[:-7]
        current_uuid += '.txt'
        file = self.directory / 'processed' / current_uuid
        with open(file, 'w+', encoding="utf-8") as f:
            f.write(document)

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

