import pandas as pd
import glob
from scipy.spatial import distance

#zscores = pd.read_hdf('2000_zscore_corpusDE.h5')

# normale Distanz/ zscores voneinander abziehen

def calculate_distance(df, unknown):
    series = []
    for index, row in df.iterrows():
        manhattan = distance.cityblock(row, df.loc[unknown])
        cosine = distance.cosine(row, df.loc[unknown])
        euclidean = distance.euclidean(row, df.loc[unknown])
        series.append(pd.Series([manhattan, cosine, euclidean, '?'], ['manhattan', 'cosine', 'euclidean', 'label'], name=index))
    return series


def create_distancedf(df, unknown):
    distance = calculate_distance(df, unknown)
    data_distance = pd.DataFrame(distance)
    data_distance.sort_values(by=['manhattan', 'cosine', 'euclidean'], inplace=True)
    data_distance = data_distance.round(2)

    return data_distance


def assign_labels(df, u):
    d = create_distancedf(df, u)
    d.name = u
    for i, row in d.iterrows():
        author = i.split(',')[0]
        d.loc[i, 'author'] = author
        if d.author[0] == d.author[i]:
            d.loc[i, 'label'] = 'same'
        else:
            d.loc[i,'label'] = 'different'
    return d

# hier ein loop mit pfad, der für alle mfw Werte die zscore Dateien einliest (alle .h5 Endungen?)
path = 'results/piperDE/zscores/*.h5'
prefix = 'results/piperDE/zscores/'
corpus = 'piperDE'
for file in glob.glob(path):
    filename = file.replace(prefix, '')
    mfw = filename.split('_')[0]

    zscores = pd.read_hdf(file)
    attribution = pd.DataFrame()
    for u in zscores.index:
        attribution = pd.concat([attribution, assign_labels(zscores, u)])

    attribution.to_csv(str(mfw) + '_delta_' + str(corpus) + '.h5')


#d.to_csv(d.name + '.csv')
