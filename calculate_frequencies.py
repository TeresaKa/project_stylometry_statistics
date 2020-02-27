from collections import Counter
import glob
import regex as re
from nltk.corpus import stopwords
import pandas as pd
from scipy.stats import zscore


def tokenize(lines, token=re.compile(r'\p{L}+')):
    for line in lines:
        yield from map(str.lower, token.findall(line))

#anderer Tokenizer ?
# def remove_stopwords(tokens):
#     return (token for token in tokenize(tokens) if token not in set(stopwords.words('german')))


def wordcounts_in_file(f_name):
    with open(f_name, encoding='utf-8') as f:
        # return Counter(remove_stopwords(f))
        return Counter(tokenize(f))


def word2freq(counts):
    words = []
    freq = []
    for c in counts:
        words.append(c)
        freq.append(counts[c])
    return words, freq


# CountVectorizer von scikit learn

def create_pd_series(path, prefix):
    series = []
    for file in glob.glob(path):
        filename = file.replace(prefix, '')
        counts = wordcounts_in_file(file)
        words, freq = word2freq(counts)
        series.append(pd.Series(freq, words, name=filename))
        print(filename)  # später löschen
    return series


path = 'dataset/corpus_DE/*.txt'
prefix = 'dataset/corpus_DE/'

series = create_pd_series(path, prefix)


def create_dataframe(series, mfw):
    df = pd.DataFrame(series)

    df = df.fillna(0)

    df = df.div(df.sum(axis=1), axis=0)

    df.loc['Total_per_word'] = df.sum()
    df = df.sort_values(by='Total_per_word', axis=1, ascending=False)
    df.drop('Total_per_word', inplace=True, axis=0)

    df.drop(df.columns[mfw:], inplace=True, axis=1)

    zscores = df.apply(zscore)

    # häufigste Wörter evtl erst nach zscore Berechnung abeschneiden (andere Mittelwerte/ Standardabweichungen)

    return df, zscores


freq, zscores = create_dataframe(series, 2000)
print(zscores.head())
print(freq)

zscores.to_hdf('corpusDE.h5', key='data', mode='w')
