from subject_classifier import SubjectClassifier
import time

directory = ""
export_to = ""

while 1:
    with open('run_file.txt', 'r+') as run_file:
        directory = run_file.readline()
        export_to = run_file.readline()
    directory = directory.rstrip()
    export_to = export_to.rstrip()
    print(directory)
    print(export_to)

    open('run_file.txt', 'w').close()

    time.sleep(3)

    if directory != "" and export_to != "":
        classifier = SubjectClassifier()
        classifier.import_data(directory)
        classifier.classify_documents()
        classifier.export_data(directory, export_to)

    directory = ""
    export_to = ""
