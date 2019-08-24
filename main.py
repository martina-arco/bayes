import csv
from bayes import compute
from bayes import TrainingDataElement

training_data = []

with open('PreferenciasBritanicos.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)
    for row in readCSV:
        attribute_list = []
        for i in range(0, 5):
            attribute_list.append(int(row[i], 2))
        training_data.append(TrainingDataElement(attribute_list, row[5]))

attributes_to_evaluate = [1, 0, 1, 1, 0]

print("Class predicted is:", compute(training_data, attributes_to_evaluate))
