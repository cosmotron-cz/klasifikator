from pathlib import Path

class Helper:
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