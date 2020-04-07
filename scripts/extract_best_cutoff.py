import pandas as pd
import glob

path = ''
corpora_best = []

for file in glob.glob(path):
    best_values = pd.read_csv('Chinese_best_cutoff_values')
    best_values.sort_values(by=['fp', 'fn'], ascending=True, inplace=True)
    corpus_best = best_values.iloc[0]
    corpora_best.append(corpus_best)
print(corpora_best)
