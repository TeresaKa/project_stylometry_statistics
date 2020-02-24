import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

zscores = pd.read_hdf('corpusDE.h5')

# normale Distanz/ zscores voneinander abziehen

def calculate_distance(df, unknown):
    series = []
    for index, row in zscores.iterrows():
        manhattan = distance.cityblock(row, zscores.loc[unknown])
       # plt.plot(i, manhattan, "r.")
        cosine = distance.cosine(row, zscores.loc[unknown])
        #plt.plot(i, cosine, "b.")
        euclidean = distance.euclidean(row, zscores.loc[unknown])
        #plt.plot(i, euclidean, "g.")
        series.append(pd.Series([manhattan, cosine, euclidean], ['manhattan', 'cosine', 'euclidean'], name=index))
    return series



l = ['UNBEKANNT.txt', 'Lewald,-Fanny_Clementine.txt']

def create_distancedf(unknown):
    distance = calculate_distance(zscores, unknown)
    data_distance = pd.DataFrame(distance)
    data_distance.sort_values(by=['manhattan', 'cosine', 'euclidean'], inplace=True)
    data_distance = data_distance.round(2)

    return data_distance


d = pd.DataFrame()
#das evtl in create_distancedf integrieren --> was ist besser? von Hand zu df zusammenfassen, aber einzelne df pro Text haben, oder gleich nur gesamtes df?
for u in l:
    d = pd.concat([d, create_distancedf(u)])
print(d)
# plt.ylabel('Distance')
# plt.xlabel('Text')
# plt.xticks(np.arange(0, 76, step=1))
# plt.show()

# data_distance.to_csv('corpusDE_distance.csv')