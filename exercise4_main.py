from bayesian_network import BayesianNetwork
from binary_file_reader import *

bayesian_network = BayesianNetwork()
bayesian_network.calculate_probabilities(variables)
rank_probabilities = bayesian_network.rank_probabilities
gpa_probabilities = bayesian_network.gpa_probabilities
gre_probabilities = bayesian_network.gre_probabilities
admit_probabilities = bayesian_network.admit_probabilities

# a

result = 0
admit = 0
rank = 1

for gre in [GRE_LESS_THAN_500, GRE_MORE_THAN_500]:
    for gpa in [GPA_LESS_THAN_3, GPA_MORE_THAN_3]:
        result += bayesian_network.get_joint_probability(admit, gpa, gre, rank)

result = result / rank_probabilities[rank]

print("a. Probability of not being admitted going to rank 1 school:", result)

# b

rank = 2
gre = GRE_LESS_THAN_500
gpa = GPA_MORE_THAN_3

denominator = 0

for admit in [0, 1]:
    denominator += bayesian_network.get_joint_probability(admit, gpa, gre, rank)

admit = 0

result = bayesian_network.get_joint_probability(admit, gpa, gre, rank) / denominator

print("b. Probability of being admitted going to rank 2 school, gre 450 and gpa 3.5:", result)


