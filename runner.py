from subject_classifier import SubjectClassifier
from trainer import Trainer
from elastic_handler import ElasticHandler
from helper.config_handler import ConfigHandler
import time
from datetime import datetime
from yagmail import SMTP


while 1:
    date_now = datetime.now()

    planned_trainings = ElasticHandler.get_planned_trainings()
    planned_trainings.sort(key=lambda p: p['date'])
    if datetime.fromisoformat(planned_trainings[0]['date']) <= date_now:
        models_dir = ConfigHandler.get_models_dir()
        ElasticHandler.update_planned_training(planned_trainings[0]['id'], None, None, None, None, 'Running', None, None)
        error = None
        model_name = "model_" + date_now.strftime("%Y_%m_%d_%H_%m")
        try:
            trainer = Trainer()
            trainer.import_data(planned_trainings[0]['data'])
            trainer.train()
            trainer.delete_index()
        except Exception as error:
            error = str(error)
            pass
        if error is not None:
            ElasticHandler.update_planned_training(planned_trainings[0]['id'], None, None, None, None, 'ERROR', error, None)
            email = planned_trainings[0].get('email', None)
            if email is not None or email != '':
                contents = [
                    "Trénovanie naplánované na " + str(planned_trainings[0]['date']) + "skončilo s chybov: ",
                    error
                ]
                yag = SMTP('klasifikator.cosmotron@gmail.com')
                yag.send(email, 'Ukončené trénovanie s chybov', contents)
                yag.close()

        else:
            ElasticHandler.update_planned_training(planned_trainings[0]['id'], None, None, None, None, 'OK', None,
                                                   models_dir + '/' + model_name)
            email = planned_trainings[0].get('email', None)
            if email is not None or email != '':
                contents = [
                    "Trénovanie naplánované na " + str(planned_trainings[0]['date']) + "skončilo OK"
                ]
                yag = SMTP('klasifikator.cosmotron@gmail.com')
                yag.send(email, 'Ukončené trénovanie OK', contents)
                yag.close()

    planned_classifications = ElasticHandler.get_unstarted_classifications()
    planned_classifications.sort(key=lambda p: p['date'])

    if datetime.fromisoformat(planned_classifications[0]['date']) <= date_now:
        export_directory = ConfigHandler.get_export_dir()
        ElasticHandler.update_planned_classification(planned_classifications[0]['id'], None, None, None, None, None,
                                                     'Running', None, None)
        error = None
        export_name = "export_" + date_now.strftime("%Y_%m_%d_%H_%m")
        try:
            classifier = SubjectClassifier(planned_classifications[0]['model'])
            classifier.import_data(planned_classifications[0]['data'])
            classifier.classify_documents()
            classifier.export_data(planned_classifications[0]['data'], export_directory + '/' + export_name)
        except Exception as error:
            error = str(error)
            pass
        if error is not None:
            ElasticHandler.update_planned_classification(planned_classifications[0]['id'], None, None, None, None, None,
                                                         'ERROR', error, None)
            email = planned_trainings[0].get('email', None)
            if email is not None or email != '':
                contents = [
                    "Klasifikácia naplánovaná na " + str(planned_trainings[0]['date']) + "skončila s chybov: ",
                    error
                ]
                yag = SMTP('klasifikator.cosmotron@gmail.com')
                yag.send(email, 'Ukončená klasifikácia s chybov', contents)
                yag.close()
        else:
            ElasticHandler.update_planned_classification(planned_classifications[0]['id'], None, None, None, None, None,
                                                         'OK', None, export_directory + '/' + export_name)
            email = planned_trainings[0].get('email', None)
            if email is not None or email != '':
                contents = [
                    "Klasifikácia naplánovaná na " + str(planned_trainings[0]['date']) + "skončila OK"
                ]
                yag = SMTP('klasifikator.cosmotron@gmail.com')
                yag.send(email, 'Ukončená klasifikácia OK', contents)
                yag.close()

