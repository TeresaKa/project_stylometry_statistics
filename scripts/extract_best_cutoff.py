import glob
import pandas as pd


def extract_best(path):
    """ Extracts best cutoff value for every corpus"""
    corpora_best = []
    entire = pd.DataFrame()
    for file in glob.glob(path):
        best_values = pd.read_csv(file)
        best_values.sort_values(by=['fp', 'fn'], ascending=True, inplace=True)
        corpus_best = best_values.iloc[0]
        corpora_best.append(corpus_best)

        all_val = best_values
        entire = pd.concat([entire, all_val])
    return entire, corpora_best


if __name__ == '__main__':
    path = '*.csv'
    entire, corpora_best = extract_best(path)

    entire.to_csv('entire_all_cutoff.csv')
    best = pd.DataFrame(corpora_best)
    best.to_csv('entire_best_cutoff.csv')
