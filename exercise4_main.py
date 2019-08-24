import csv
from bayesian_network import BayesianNetwork

ADMIT = 0
GRE = 1
GPA = 2
RANK = 3

GRE_MORE_THAN_500 = 1
GRE_LESS_THAN_500 = 0

GPA_MORE_THAN_3 = 1
GPA_LESS_THAN_3 = 0

variables = []


class Variables:
    def __init__(self, rank1, gre1, gpa1, admit1):
        self.rank = rank1
        self.gre = gre1
        self.gpa = gpa1
        self.admit = admit1

    def __str__(self):
        return str(self.rank) + \
               (", Less than 500" if self.gre == GRE_LESS_THAN_500 else ", More than 500") + \
               (", Less than 3" if self.gpa == GPA_LESS_THAN_3 else ", More than 3") + \
               ", " + str(self.admit)


with open("binary(1).csv", encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)
    count = 0
    for line in csv_reader:
        if len(line) == 4:
            rank = int(line[RANK])
            gre = GRE_MORE_THAN_500 if float(line[GRE]) >= 500 else GRE_LESS_THAN_500
            gpa = GPA_MORE_THAN_3 if float(line[GPA]) >= 3 else GPA_LESS_THAN_3
            admit = int(line[ADMIT])
            variables.append(Variables(rank, gre, gpa, admit))

bayesian_network = BayesianNetwork()
bayesian_network.calculate_probabilities(variables)
rank_probabilities = bayesian_network.rank_probabilities
gpa_probabilities = bayesian_network.gpa_probabilities
gre_probabilities = bayesian_network.gre_probabilities
admit_probabilities = bayesian_network.admit_probabilities

# punto a

result = 0
admit = 0
rank = 1

for gre in [GRE_LESS_THAN_500, GRE_MORE_THAN_500]:
    for gpa in [GPA_LESS_THAN_3, GPA_MORE_THAN_3]:
        result += bayesian_network.get_joint_probability(admit, gpa, gre, rank)

result = result / rank_probabilities[rank]

print("Probabilidad de que no sea admitido a la universidad yendo a escuela de rango 1:", result)

# punto b

rank = 2
gre = GRE_LESS_THAN_500
gpa = GPA_MORE_THAN_3

denominator = 0

for admit in [0, 1]:
    denominator += bayesian_network.get_joint_probability(admit, gpa, gre, rank)

admit = 0

result = bayesian_network.get_joint_probability(admit, gpa, gre, rank) / denominator

print("Probabilidad de que sea admitida con rango 2, gre 450 y gpa 3.5:", result)


