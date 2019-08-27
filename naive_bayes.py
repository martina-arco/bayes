class NaiveBayes:

    def __init__(self):
        self.classification_probabilities = {}
        self.classification_sizes = {}
        self.total_attribute_values = 0
        self.attribute_probabilities = {}

    def set_probabilities(self, classification_probabilities, attribute_probabilities):
        self.classification_probabilities = classification_probabilities
        self.attribute_probabilities = attribute_probabilities

    def train(self, training_data, total_attribute_values):
        classification_frequencies = {}
        attribute_frequencies = {}
        total_amount = 0

        for training_data_element in training_data:
            classification = training_data_element.classification
            classification_frequency = classification_frequencies.get(classification)

            if classification_frequency is not None:
                classification_frequency += 1
                classification_frequencies[classification] = classification_frequency
            else:
                classification_frequencies[classification] = 1

            for i in range(0, len(training_data_element.attributes)):
                attribute_frequency = attribute_frequencies.get(training_data_element.attributes[i])
                if attribute_frequency is not None:
                    frequency = attribute_frequency.get(classification)
                    if frequency is not None:
                        attribute_frequency[classification] += 1
                    else:
                        attribute_frequency[classification] = 1
                else:
                    attribute_frequencies[training_data_element.attributes[i]] = {classification: 1}

            total_amount += 1

        classification_probabilities = {}
        for classification, frequency in classification_frequencies.items():
            classification_probabilities[classification] = frequency/total_amount

        for attribute in attribute_frequencies:
            frequencies = attribute_frequencies[attribute]
            for classification, value in frequencies.items():
                attribute_frequencies[attribute][classification] = \
                    (value + 1) / (classification_frequencies[classification] + total_attribute_values)

        self.classification_probabilities = classification_probabilities
        self.attribute_probabilities = attribute_frequencies
        self.total_attribute_values = total_attribute_values
        self.classification_sizes = classification_frequencies

    def classify(self, data):

        probabilities = self.classification_probabilities.copy()

        for item in data:
            attribute = self.attribute_probabilities.get(item)

            for classification in probabilities:
                if attribute is not None and attribute.get(classification) is not None:
                    probability = attribute[classification]
                else:
                    probability = 1 / (self.classification_sizes[classification] + self.total_attribute_values)

                probabilities[classification] *= probability
        most_likely = max(probabilities, key=probabilities.get)
        return most_likely, probabilities[most_likely]

    # matrix[prediction][actual]
    def get_confusion_matrix(self, testing_data):

        matrix = {}
        for classification in self.classification_sizes.keys():
            matrix[classification] = {}
            for other in self.classification_sizes.keys():
                matrix[classification][other] = 0

        for test in testing_data:
            most_likely = self.classify(test.attributes)
            matrix[most_likely[0]][test.classification] += 1

        return matrix

    def get_accuracy(self, confusion_matrix):
        accuracy = 0
        denominator = 0

        for classification in confusion_matrix.keys():
            accuracy += confusion_matrix[classification][classification]
            denominator += confusion_matrix[classification][classification]
            for other, value in confusion_matrix[classification].items():
                if other != classification:
                    denominator += value

        return accuracy/denominator

    def get_precision(self, confusion_matrix):
        precision = {}

        for classification in confusion_matrix.keys():
            precision[classification] = confusion_matrix[classification][classification]
            denominator = confusion_matrix[classification][classification]
            for other, value in confusion_matrix[classification].items():
                if other != classification:
                    denominator += value
            precision[classification] /= denominator
        return precision

    def get_recall(self, confusion_matrix):
        recall = {}

        for classification in confusion_matrix.keys():
            recall[classification] = confusion_matrix[classification][classification]
            denominator = confusion_matrix[classification][classification]
            for other, value in confusion_matrix.items():
                if other != classification:
                    denominator += value[classification]
            recall[classification] /= denominator
        return recall

    def get_f1(self, confusion_matrix):
        p = self.get_precision(confusion_matrix)
        r = self.get_recall(confusion_matrix)
        f1 = {}

        for classification in p:
            f1[classification] = (2 * p[classification] * r[classification]) / (p[classification] + r[classification])
        return f1

    def true_positive(self, confusion_matrix):
        return self.get_recall(confusion_matrix)


    def false_positive(self, confusion_matrix):
        fp = {}

        for classification in confusion_matrix.keys():
            fp[classification] = confusion_matrix[classification][classification]
            denominator = confusion_matrix[classification][classification]
            for other in confusion_matrix[classification].keys():
                if other != classification:
                    denominator += confusion_matrix[other][other]
            fp[classification] /= denominator
        return fp
