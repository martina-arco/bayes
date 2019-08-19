class AttributeClassification:
    def __init__(self, attribute, classification):
        self.attibute = attribute
        self.classification = classification

    def __str__(self) -> str:
        return str(self.attibute) + " " + str(self.classification)


def compute(training_data, attributes_to_evaluate):
    classification_frequencies = calculate_classification_relative_frequencies(training_data)

    attributes_frequencies = calculate_attributes_frequencies(training_data, attributes_to_evaluate,
                                                              classification_frequencies)

    return calculate_vnb(classification_frequencies, attributes_frequencies)


def calculate_classification_relative_frequencies(training_data):
    classification_frequencies = {}
    total_amount = 0

    for training_data_element in training_data:
        classification = training_data_element.classification
        classification_frequency = classification_frequencies.get(classification)

        if classification_frequency is not None:
            classification_frequency += 1
            classification_frequencies[classification] = classification_frequency
        else:
            classification_frequencies[classification] = 1

        total_amount += 1

    for classification in classification_frequencies:
        relative_frequency = classification_frequencies.get(classification) / total_amount
        classification_frequencies[classification] = relative_frequency

    return classification_frequencies


def calculate_attributes_frequencies(training_data, attributes_to_evaluate, classification_frequencies):
    attributes_frequencies = {}
    attribute_amount = len(attributes_to_evaluate)

    for classification in classification_frequencies:
        attribute_frequency_for_classification = [0] * attribute_amount
        attributes_total = [0] * attribute_amount

        for training_data_element in training_data:

            if training_data_element.classification == classification:

                for i in range(0, attribute_amount):
                    attributes_total[i] += 1
                    if attributes_to_evaluate[i] == training_data_element.attributes[i]:
                        attribute_frequency_for_classification[i] += 1

        for i in range(0, attribute_amount):
            attribute_classification_pair = AttributeClassification(attributes_to_evaluate[i], classification)
            attributes_frequencies[attribute_classification_pair] = \
                attribute_frequency_for_classification[i] / attributes_total[i]

    return attributes_frequencies


def calculate_vnb(classification_frequencies, attributes_frequencies):
    probabilities = classification_frequencies.copy()

    for attribute_classification_pair in attributes_frequencies:

        for classification in classification_frequencies:

            if attribute_classification_pair.classification == classification:

                probabilities[classification] *= attributes_frequencies[attribute_classification_pair]

    return max(probabilities, key=probabilities.get)
