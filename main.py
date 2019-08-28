import csv
import random

from naive_bayes import TrainingDataElement
from naive_bayes import NaiveBayes
from text_classifier import data as text_classifier_data

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

                ############################# EXERCISE 1 #############################
probabilities = {("P1", 1): {"Joven": 0.95, "Viejo": 0.03}, ("P2", 1): {"Joven": 0.05, "Viejo": 0.82},
                 ("P3", 1): {"Joven": 0.02, "Viejo": 0.34}, ("P4", 1): {"Joven": 0.2, "Viejo": 0.92},
                 ("P1", 0): {"Joven": 0.05, "Viejo": 0.87}, ("P2", 0): {"Joven": 0.95, "Viejo": 0.18},
                 ("P3", 0): {"Joven": 0.98, "Viejo": 0.66}, ("P4", 0): {"Joven": 0.8, "Viejo": 0.08}}
old_probability = 0.9

print("EXERCISE 1")
data_young_probability = probabilities[("P1", 1)]["Joven"] * probabilities[("P2", 0)]["Joven"] \
                         * probabilities[("P3", 1)]["Joven"] * probabilities[("P4", 0)]["Joven"] * (1 - old_probability)
data_old_probability = probabilities[("P1", 1)]["Viejo"] * probabilities[("P2", 0)]["Viejo"] \
                         * probabilities[("P3", 1)]["Viejo"] * probabilities[("P4", 0)]["Viejo"] * old_probability
print("Young viewer probability = {0:.3f}\n".format(data_young_probability / (data_young_probability + data_old_probability)))
print("Old viewer probability = {0:.3f}\n".format(data_old_probability / (data_young_probability + data_old_probability)))

                ############################# EXERCISE 2 #############################
nb = NaiveBayes()
nb.train(training_data2, 2)
most_likely = nb.classify(attributes_to_evaluate2)
print("EXERCISE 2")
print("Exercise 2, class predicted is: {0} with probability {1:.3f}%\n".format(most_likely[0], most_likely[1] * 100))

                ############################# EXERCISE 3 #############################

testing = random.sample(range(0, len(text_classifier_data) - 1), int(len(text_classifier_data) * 0.1))
testing_data = []
for index in sorted(testing, reverse=True):
    testing_data.append(text_classifier_data.pop(index))

nb = NaiveBayes()
nb.train(text_classifier_data, 2)
matrix = nb.get_confusion_matrix(testing_data)
accuracy = nb.get_accuracy(matrix)
precision = nb.get_precision(matrix)
recall = nb.get_recall(matrix)
f1 = nb.get_f1(matrix)
tp = nb.true_positive(matrix)
fp = nb.false_positive(matrix)

print("EXERCISE 3")
print("Accuracy: ", accuracy)

print("\nPrecision")
for classification, value in precision.items():
    print("Precision for class", classification, ": ", value)

print("\nRecall")
for classification, value in recall.items():
    print("Recall for class", classification, ": ", value)

print("\nF1-score")
for classification, value in f1.items():
    print("F1 for class", classification, ": ", value)

print("\nTrue positives")
for classification, value in tp.items():
    print("True positive for class", classification, ": ", value)

print("\nFalse positives")
for classification, value in fp.items():
    print("False positive for class", classification, ": ", value)

print("\nConfusion matrix")
print('{:15.15}'.format(' '), end=' ')
string = ' '.join(['{:^15.15}'.format(item) for item in matrix.keys()])
print(string)

for classification in matrix.items():
    print('{:15.15}'.format(classification[0]), end="")
    string = ' '.join(['{:^15}'.format(item) for item in classification[1].values()])
    print(string)