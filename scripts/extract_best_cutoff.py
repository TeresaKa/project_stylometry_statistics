import pandas as pd
import glob

path = 'results/all_cutoffs/*.csv'
corpora_best = []
alle = []
for file in glob.glob(path):
    best_values = pd.read_csv(file)
    best_values.sort_values(by=['fp', 'fn'], ascending=True, inplace=True)
    alle.append(best_values)
    corpus_best = best_values.iloc[0]
    corpora_best.append(corpus_best)
entire = pd.DataFrame(alle)
#entire.to_csv('entire_all_cutoff.csv')
best = pd.DataFrame(corpora_best)
print(best)
#best.to_csv('entire_best_cutoff.csv')
