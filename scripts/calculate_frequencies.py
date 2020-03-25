from collections import Counter
import glob
import regex as re
from nltk.corpus import stopwords
import pandas as pd
from scipy.stats import zscore


def tokenize(lines, token=re.compile(r'\p{L}+')):
    """

    :param lines (str): object to be tokenized, e.g. file
    :param token: pattern to tokenize lines
    :return: lowered and tokenized string
    """
    for line in lines:
        yield from map(str.lower, token.findall(line))

#anderer Tokenizer ?
# def remove_stopwords(tokens):
#     return (token for token in tokenize(tokens) if token not in set(stopwords.words('german')))


def wordcounts_in_file(f_name):
    """

    :param f_name: filename of file to be analyzed
    :return: Counter of tokenized file
    """
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


def create_pd_series(path, prefix):
    series = []
    for file in glob.glob(path):
        filename = file.replace(prefix, '')
        counts = wordcounts_in_file(file)
        words, freq = word2freq(counts)
        series.append(pd.Series(freq, words, name=filename))
        print(filename)  # später löschen
    return series


path = 'dataset/piperDE/*.txt'
prefix = 'dataset/piperDE/'

series = create_pd_series(path, prefix)


def create_dataframe(series, mfw):
    df = pd.DataFrame(series)

    df = df.fillna(0)

    df = df.div(df.sum(axis=1), axis=0)

    df.loc['Total_per_word'] = df.sum()
    df = df.sort_values(by='Total_per_word', axis=1, ascending=False)
    df.drop('Total_per_word', inplace=True, axis=0)

    zscores = df.apply(zscore)

    zscores.drop(zscores.columns[mfw:], inplace=True, axis=1)

    return df, zscores


mfw_values = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
corpus = 'piperDE'
for mfw in mfw_values:
    freq, zscores = create_dataframe(series, mfw)
    zscores.to_hdf(str(mfw) + '_zscore_' + str(corpus) + '.h5', key='data', mode='w')
