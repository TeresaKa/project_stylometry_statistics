from collections import Counter
import glob
import regex as re
from nltk.corpus import stopwords
import pandas as pd
from scipy.stats import zscore


def tokenize(lines, pattern=re.compile(r'\p{L}+')):
    """

    :param lines: object to be tokenized, e.g. file
    :param pattern: pattern to tokenize 'lines'
    :return: lowered and tokenized string
    """
    for line in lines:
        yield from map(str.lower, pattern.findall(line))

#anderer Tokenizer ?
# def remove_stopwords(tokens):
#     return (token for token in tokenize(tokens) if token not in set(stopwords.words('german')))


def wordcounts_in_file(file):
    """

    :param file: filename of file to be analyzed
    :return: Counter of tokenized file
    """
    with open(file, encoding='utf-8') as f:
        # return Counter(remove_stopwords(f))
        return Counter(tokenize(f))


def word2freq(counts):
    """

    :param counts: a Counter from a file to be analyzed
    :return: list of each words and their frequencies
    """
    words = []
    freq = []
    for c in counts:
        words.append(c)
        freq.append(counts[c])
    return words, freq


def create_pd_series(path, prefix):
    """

    :param path: path to files
    :param prefix: prefix to remove from filename for further use in Series
    :return: pd.Series with words and wordcounts
    """
    freq_list = []
    for file in glob.glob(path):
        filename = file.replace(prefix, '')
        counts = wordcounts_in_file(file)
        words, freq = word2freq(counts)
        freq_list.append(pd.Series(freq, words, name=filename))
        print(filename)  # später löschen
    return freq_list


def create_dataframe(series, mfw):
    """

    :param series: list of pd.Series, created e.g. with create_pd_series()
    :param mfw: value of most frequent words
    :return: document-term-matrix as pd.Dataframe with zscores
    """
    df = pd.DataFrame(series)
    df = df.fillna(0)

    df = df.div(df.sum(axis=1), axis=0)

    df.loc['Total_per_word'] = df.sum()
    df = df.sort_values(by='Total_per_word', axis=1, ascending=False)
    df.drop('Total_per_word', inplace=True, axis=0)

    zscores = df.apply(zscore)
    zscores.drop(zscores.columns[mfw:], inplace=True, axis=1)
    return df, zscores


path = 'dataset/corpusDE/*.txt'
prefix = 'dataset/corpusDE/'

series = create_pd_series(path, prefix)

#mfw_values = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
mfw_values = [50]
corpus = 'corpusDE'
for mfw in mfw_values:
    freq, zscores = create_dataframe(series, mfw)
    zscores.to_hdf(str(mfw) + '_zscore_' + str(corpus) + '.h5', key='data', mode='w')
    #freq.to_hdf(str(mfw) + '_rel_freq_' + str(corpus) + '.h5', key='data', mode='w')
