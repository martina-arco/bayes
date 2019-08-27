import csv
from naive_bayes import TrainingDataElement

DATE = 0
TITLE = 1
PAPER = 2
CLASSIFICATION = 3

WORDS_TO_IGNORE = {"LA", "LAS", "LOS", "LO", "EL", "ELLA", "ELLOS", "ELLAS", "Y", "A", "CON", "DEL", "POR", "SE", "UN", "UNA", "COMO",
                   "DE", "ESTE", "ESTOS", "ESTAS", "EN", "TU", "TUS", "PARA", "SU", "SUS", "AL", "NO", "SI", "SON", "ES", "QUE", "HACIA",
                   "DONDE"}

data = []
different_words_average = 0

with open("aa_bayes.tsv", encoding='utf-8') as tsvfile:
    tsv_reader = csv.reader(tsvfile, delimiter="\t")
    next(tsv_reader)
    for line in tsv_reader:
        if len(line) == 4:
            words = line[TITLE].upper().split()
            words = list(map(lambda word: (word, 1), filter(lambda word: word not in WORDS_TO_IGNORE, words)))
            classification = line[CLASSIFICATION]
            if classification != "Destacadas" and classification != "Noticias destacadas":
                data.append(TrainingDataElement(words, classification))
