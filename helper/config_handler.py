import configparser
import os


class ConfigHandler:
    @staticmethod
    def get_config():
        config = configparser.ConfigParser()
        directory = os.path.dirname(os.path.realpath(__file__))
        config.read(directory + '/../config.ini')
        return config

    @staticmethod
    def get_train_data_dir():
        config = ConfigHandler.get_config()
        train_data_dir = config.get('data', 'train_data')
        return train_data_dir

    @staticmethod
    def get_class_data_dir():
        config = ConfigHandler.get_config()
        class_data_dir = config.get('data', 'class_data')
        return class_data_dir

    @staticmethod
    def get_export_dir():
        config = ConfigHandler.get_config()
        export_dir = config.get('data', 'export')
        return export_dir

    @staticmethod
    def get_models_dir():
        config = ConfigHandler.get_config()
        export_dir = config.get('data', 'models')
        return export_dir
