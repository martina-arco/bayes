class BayesianNetwork:

    def __init__(self):
        self.rank_probabilities = {}
        self.gpa_probabilities = {}
        self.gre_probabilities = {}
        self.admit_probabilities = {}
        self.total_amount_rank_gpa_gre = {}

    def calculate_probabilities(self, variables):
        total_amount = 0
        rank_frequencies = {}
        gpa_frequencies = {}
        gre_frequencies = {}
        admit_frequencies = {}

        for variable in variables:
            rank = variable.rank
            gpa = variable.gpa
            gre = variable.gre
            admit = variable.admit

            rank_frequency = rank_frequencies.get(rank, 0)
            rank_frequency += 1
            rank_frequencies[rank] = rank_frequency

            set_conditional_probability(gpa_frequencies, rank, gpa)
            set_conditional_probability(gre_frequencies, rank, gre)
            set_conditional_probability(admit_frequencies, (rank, gpa, gre), admit)

            total_amount_tuples = self.total_amount_rank_gpa_gre.get((rank, gpa, gre), 0)
            total_amount_tuples += 1
            self.total_amount_rank_gpa_gre[(rank, gpa, gre)] = total_amount_tuples

            total_amount += 1

        for frequency in rank_frequencies.items():
            self.rank_probabilities[frequency[0]] = frequency[1] / total_amount

        calculate_probability(gpa_frequencies, rank_frequencies, self.gpa_probabilities, 2)
        calculate_probability(gre_frequencies, rank_frequencies, self.gre_probabilities, 2)
        calculate_probability(admit_frequencies, self.total_amount_rank_gpa_gre, self.admit_probabilities, 2)

    def get_joint_probability(self, admit, gpa, gre, rank):
        return self.rank_probabilities[rank] * self.gpa_probabilities[(gpa, rank)] * self.gre_probabilities[(gre, rank)] * \
                 self.get_admit_probability(admit, gpa, gre, rank)

    def get_admit_probability(self, admit, gpa, gre, rank):
        probability = self.admit_probabilities.get((admit, (rank, gpa, gre)))

        if probability is None:
            probability = 1 / (self.total_amount_rank_gpa_gre.get((rank, gpa, gre), 0) + 2)

        return probability


def set_conditional_probability(frequencies, independent_variables, dependant_variable):
    dependant_variable_frequency = frequencies.get(dependant_variable)

    if dependant_variable_frequency is not None:
        frequency = dependant_variable_frequency.get(independent_variables)

        if frequency is not None:
            dependant_variable_frequency[independent_variables] += 1
        else:
            dependant_variable_frequency[independent_variables] = 1

    else:
        frequencies[dependant_variable] = {independent_variables: 1}


def calculate_probability(frequencies, independent_variables_frequency, probabilities, total_attribute_values):
    for item in frequencies.items():
        dependant_variable = item[0]
        for frequencies in item[1].items():
            independent_variable = frequencies[0]
            probabilities[(dependant_variable, independent_variable)] = \
                (frequencies[1] + 1) / (independent_variables_frequency[independent_variable] + total_attribute_values)
