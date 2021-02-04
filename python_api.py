import os

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import json
from helper.config_handler import ConfigHandler
from elastic_handler import ElasticHandler
from elasticsearch import ElasticsearchException


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def return_error(message):
    result = {'code': -1, 'message': message, 'success': False}
    return json.dumps(result)


def elastic_error():
    return return_error("Chyba databázy")


def internal_error():
    return return_error("Interná chyba serveru")


def return_ok(data=None):
    result = {'code': 0, 'message': 'OK', 'success': True}
    if data is not None:
        result['data'] = data
    return json.dumps(result)


@app.route('/get_planned_trainings')
@cross_origin()
def get_planned_trainings():
    try:
        results = ElasticHandler.get_planned_trainings()
    except ElasticsearchException:
        return elastic_error()
    return return_ok(results)


@app.route('/new_planned_training', methods=['POST'])
@cross_origin()
def new_planned_training():
    data = request.json.get('data', None)
    note = request.json.get('note', '')
    date = request.json.get('date', None)
    email = request.json.get('email', '')
    if data is None or date is None:
        return return_error("Chýbajú dáta, alebo dátum spustenia pre plánované trénovanie")
    try:
        ElasticHandler.create_planned_training(data, note, date, email)
    except ElasticsearchException:
        return elastic_error()
    return return_ok(None)


@app.route('/delete_planned_training', methods=['POST'])
@cross_origin()
def delete_planned_training():
    training_id = request.json.get('id', None)
    if training_id is None:
        return return_error("Chýba id plánovaného trénovania")
    try:
        ElasticHandler.delete_planned_training(training_id)
    except ElasticsearchException:
        return elastic_error()
    return return_ok(None)


@app.route('/update_planned_training', methods=['POST'])
@cross_origin()
def update_planned_training():
    training_id = request.json.get('id', None)
    if training_id is None:
        return return_error("Chýba id plánovaného trénovania")
    data = request.json.get('data', None)
    note = request.json.get('note', None)
    date = request.json.get('date', None)
    email = request.json.get('email', None)
    # try:
    ElasticHandler.update_planned_training(training_id, data, note, date, email, None, None, None)
    # except ElasticsearchException:
    #     return elastic_error()
    return return_ok(None)


@app.route('/get_training_data')
@cross_origin()
def get_training_data():
    training_data_dir = ConfigHandler.get_train_data_dir()
    subdirectories = [f.path for f in os.scandir(training_data_dir) if f.is_dir()]
    print(subdirectories)
    valid_directories = []
    for subdir in subdirectories:
        directory_contents = os.listdir(subdir)
        if 'text' in directory_contents and 'sorted_pages' in directory_contents and \
                'sloucena_id' in directory_contents and 'metadata.xml' in directory_contents:
            valid_directories.append({'data': os.path.basename(subdir)})
    print(valid_directories)
    return return_ok(valid_directories)


@app.route('/get_planned_classifications')
@cross_origin()
def get_planned_classifications():
    try:
        results = ElasticHandler.get_planned_classifications()
    except ElasticsearchException:
        return elastic_error()
    return return_ok(results)


@app.route('/new_planned_classification', methods=['POST'])
@cross_origin()
def new_planned_classification():
    data = request.json.get('data', None)
    print(data)
    model = request.json.get('model', None)
    print(model)
    note = request.json.get('note', '')
    date = request.json.get('date', None)
    email = request.json.get('email', '')
    if data is None or date is None or model is None:
        return return_error("Chýbajú dáta, model, alebo dátum spustenia pre plánovanú klasifikáciu")
    try:
        ElasticHandler.create_planned_classification(data, model, note, date, email)
    except ElasticsearchException:
        return elastic_error()
    return return_ok(None)


@app.route('/delete_planed_classification', methods=['POST'])
@cross_origin()
def delete_planed_classification():
    classification_id = request.json.get('id', None)
    if classification_id is None:
        return return_error("Chýba id plánovanej klasifikácie")
    try:
        ElasticHandler.delete_planned_classification(classification_id)
    except ElasticsearchException:
        return elastic_error()
    return return_ok(None)


@app.route('/update_planned_classification', methods=['POST'])
@cross_origin()
def update_planned_classification():
    classification_id = request.json.get('id', None)
    if classification_id is None:
        return return_error("Chýba id plánovanej klasifikácie")
    data = request.json.get('data', None)
    model = request.json.get('model', None)
    note = request.json.get('note', None)
    date = request.json.get('date', None)
    email = request.json.get('email', None)
    try:
        ElasticHandler.update_planned_classification(classification_id, data, model, note, date, email, None, None,
                                                     None)
    except ElasticsearchException:
        return elastic_error()
    return return_ok(None)


@app.route('/get_classification_data')
@cross_origin()
def get_classification_data():
    training_data_dir = ConfigHandler.get_class_data_dir()
    subdirectories = [f.path for f in os.scandir(training_data_dir) if f.is_dir()]
    print(subdirectories)
    valid_directories = []
    for subdir in subdirectories:
        directory_contents = os.listdir(subdir)
        if 'text' in directory_contents and 'sorted_pages' in directory_contents and \
                'sloucena_id' in directory_contents and 'metadata.xml' in directory_contents:
            valid_directories.append({'data': os.path.basename(subdir)})
    return return_ok(valid_directories)


@app.route('/get_models')
@cross_origin()
def get_models():
    models_dir = ConfigHandler.get_models_dir()
    subdirectories = [f.path for f in os.scandir(models_dir) if f.is_dir()]
    valid_directories = []
    for subdir in subdirectories:
        directory_contents = os.listdir(subdir)
        if 'fulltext' in directory_contents and 'keywords' in directory_contents:
            fulltext_directory = os.listdir(subdir + "/fulltext")
            keywords_directory = os.listdir(subdir + "/keywords")
            if 'category.pickle' in fulltext_directory and 'groups_labels.pickle' in fulltext_directory and \
                    'category.pickle' in keywords_directory and 'groups_labels.pickle' in keywords_directory and \
                    'tfidf.pickle' in keywords_directory:
                missing_model = False
                for i in range(1, 27):
                    if 'groups_' + str(i) + '.pickle' not in fulltext_directory or \
                            'groups_' + str(i) + '.pickle' not in keywords_directory:
                        missing_model = True
                        break
                if not missing_model:
                    valid_directories.append({'model': os.path.basename(subdir)})
    return return_ok(valid_directories)


if __name__ == '__main__':
    # app.run(host= '0.0.0.0', port=5000, debug=False, ssl_context=('cert.pem', 'key.pem'))
    app.run(host='0.0.0.0', port=5001, debug=False)
