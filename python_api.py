from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import json


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# # Celery configuration
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
#
# # Initialize Celery
# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)


# @celery.task
# def async_run_classification(directory, export_to):
#     """Spustenie klasifikacie na pozadi"""
#     print("Runnig classification")
#     # classifier = SubjectClassifier()
#     # classifier.import_data(directory)
#     # classifier.classify_documents()
#     # classifier.export_data(directory, export_to)

@app.route('/run_classification')
@cross_origin()
def run_classification():
    directory = request.args.get('directory')
    export_to = request.args.get('export_to')
    callback = request.args.get('callback')
    print(directory)
    print(export_to)
    with open('run_file.txt', 'w') as run_file:
        run_file.write(directory + '\n')
        run_file.write(export_to + '\n')
    return callback + '(' + json.dumps({"ret_code": "0", "ret_msg": ""}) + ')'

if __name__ == '__main__':
    # app.run(host= '0.0.0.0', port=5000, debug=False, ssl_context=('cert.pem', 'key.pem'))
    app.run(host='0.0.0.0', port=5001, debug=False)
