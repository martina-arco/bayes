import csv

ADMIT = 0
GRE = 1
GPA = 2
RANK = 3

GRE_MORE_THAN_500 = 1
GRE_LESS_THAN_500 = 0

GPA_MORE_THAN_3 = 1
GPA_LESS_THAN_3 = 0


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


variables = []

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
