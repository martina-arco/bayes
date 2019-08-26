import csv
import random

from bayes import compute
from bayes import TrainingDataElement
from naive_bayes import NaiveBayes
from text_classifier import training_data as training_data3

training_data = []
training_data2 = []

with open('PreferenciasBritanicos.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    attributes = next(readCSV, None)
    for row in readCSV:
        attribute_list = []
        attribute_list2 = []
        for i in range(0, 5):
            attribute_list.append(int(row[i], 2))
            attribute_list2.append((attributes[i], int(row[i], 2)))
        training_data.append(TrainingDataElement(attribute_list, row[5]))
        training_data2.append(TrainingDataElement(attribute_list2, row[5]))

attributes_to_evaluate = [1, 0, 1, 1, 0]
attributes_to_evaluate2 = [(attributes[0], 1), (attributes[1], 0), (attributes[2], 1), (attributes[3], 1), (attributes[4], 0)]

print("Class predicted is:", compute(training_data, attributes_to_evaluate))


                ############################# EXERCISE 1 #############################
probabilities = {("P1", 1): {"Joven": 0.95, "Viejo": 0.03}, ("P2", 1): {"Joven": 0.05, "Viejo": 0.82},
                 ("P3", 1): {"Joven": 0.02, "Viejo": 0.34}, ("P4", 1): {"Joven": 0.2, "Viejo": 0.92},
                 ("P1", 0): {"Joven": 0.05, "Viejo": 0.87}, ("P2", 0): {"Joven": 0.95, "Viejo": 0.18},
                 ("P3", 0): {"Joven": 0.98, "Viejo": 0.66}, ("P4", 0): {"Joven": 0.8, "Viejo": 0.08}}

nb = NaiveBayes()
nb.set_probabilities({"Joven": 0.1, "Viejo": 0.9}, probabilities)
most_likely = nb.classify([("P1", 1), ("P3", 1), ("P2", 0), ("P4", 0)])
print("Exercise 1, class predicted is: {0} with probability {1:.3f}%".format(most_likely[0], most_likely[1] * 100))

                ############################# EXERCISE 2 #############################
nb = NaiveBayes()
nb.train(training_data2, 2)
most_likely = nb.classify(attributes_to_evaluate2)
print("Exercise 2, class predicted is: {0} with probability {1:.3f}%".format(most_likely[0], most_likely[1] * 100))

                ############################# EXERCISE 3 #############################

testing = random.sample(range(0, len(training_data3) - 1), int(len(training_data3) * 0.1))
testing_data = []
for index in sorted(testing, reverse=True):
    testing_data.append(training_data3.pop(index))

nb = NaiveBayes()
nb.train(training_data3, 2)
matrix = nb.get_confusion_matrix(testing_data)
