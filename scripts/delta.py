# -*- coding: utf-8 -*-
import pandas as pd
import glob
from scipy.spatial import distance

# zscores = pd.read_hdf('2000_zscore_corpusDE.h5')

class Delta:
    def __init__(self, df, unknown):
        """

        :param df: document-term-matrix with zscores
        :param unknown: specify unknown document in 'df' to be compared with remaining texts
        """
        self.df = df
        self.unknown = unknown

    def calculate_distance(self):
        """ Calculates Manhattan, Cosine and Euclidean Delta measures and returns them as pd.Series """
        series_list = []
        for index, row in self.df.iterrows():
            manhattan = distance.cityblock(row, self.df.loc[self.unknown])
            cosine = distance.cosine(row, self.df.loc[self.unknown])
            euclidean = distance.euclidean(row, self.df.loc[self.unknown])
            series_list.append(pd.Series([manhattan, cosine, euclidean, '?'], ['manhattan', 'cosine', 'euclidean', 'label'], name=index))
        return series_list

    def create_distance_df(self):
        distance_measures = self.calculate_distance()
        distance = pd.DataFrame(distance_measures)
        distance.sort_values(by=['manhattan', 'cosine', 'euclidean'], inplace=True)  # ist das nötig?
        distance = distance.round(2)

        return distance

    def assign_labels(self):
        """ Compares author of 'unknown' text with authors of remaining texts.
        Assigns labels: 'same' if authors match, 'different' otherwise. """
        delta = self.create_distance_df()
        delta.name = self.unknown
        for i, row in delta.iterrows():
            author = i.split(',')[0]
            delta.loc[i, 'author'] = author
            if delta.author[0] == delta.author[i]:
                delta.loc[i, 'label'] = 'same'
            else:
                delta.loc[i, 'label'] = 'different'
        return delta

# hier ein loop mit pfad, der für alle mfw Werte die zscore Dateien einliest (alle .h5 Endungen?)
path = 'results/Chinese/zscores/*.h5'
prefix = 'results/Chinese/zscores/'
corpus = 'Chinese'
for file in glob.glob(path):
    filename = file.replace(prefix, '')
    mfw = filename.split('_')[0]

    zscores = pd.read_hdf(file)
    attribution = pd.DataFrame()
    for u in zscores.index:
        attribution = pd.concat([attribution, Delta(zscores, u).assign_labels()])
    attribution.to_hdf(str(mfw) + '_delta_' + str(corpus) + '.h5',  key='data', mode='w')


# d.to_csv(d.name + '.csv')
