# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import sklearn
print(sklearn.__version__)


# ### mathematische Grundlagen

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid


x = np.linspace(-10, 10, 100)
x

# +
z = sigmoid(x)

ax = plt.plot(x, z)
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 

plt.show() 

# +
#Ableitung der Sigmoid
ds = (np.exp(-x))/((1+np.exp(-x))**2)
ax = plt.plot(x, ds)
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 

plt.show() 
print(max(ds))
# -

# ### load calculated delta measures

data = pd.DataFrame(pd.read_csv('DE_attribution', index_col=0))
data = data[data.cosine != 0.00]
data

# +
x = data['cosine'].values.reshape(-1,1)
z = sigmoid(x)
ax = plt.plot(x, z)
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 

plt.show() 

# +
d = data.copy()
d.drop('manhattan', inplace=True, axis=1)
d.drop('euclidean', inplace=True, axis=1)
d.drop('author', inplace=True, axis=1)

cosine = d['cosine'].values.reshape(-1,1)  #np.unique machen?in Schleife unten werden trotzdem alle Zeilen erfasst damit
minx = min(cosine)
d
# -

cos = np.unique(cosine)
minx

# ### determine same and different authors in 1%-steps of cosine values

# +
perc = 1.1
cos_range = max(cos)-min(cos)
n = np.arange(0.01, 1.01, 0.01)

for perc in n:
    for c in cos:
        if c <= minx + cos_range*perc:
            d.loc[d.cosine==float(c), np.around(perc, decimals=2)] = 'same'
        else:
             d.loc[d.cosine==float(c), np.around(perc, decimals=2)] = 'different'
d
# -

d[d.cosine<=0.22]


# ### calculate and visualize apha and beta errors

def normalized_cnf_matrix(cls, column, mfw):
    cnf = confusion_matrix(true_label, cols, normalize='all')
    fig, ax = plt.subplots(figsize=(5,5))
    #plt.figure(figsize=(5,5))  
    sns.heatmap(cnf, annot=True, cmap=sns.color_palette("Blues"), ax = ax); #annot=True to annotate cells

    
    tn, fp, fn, tp = cnf.ravel()
    
    error_dic = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(cls)
    ax.yaxis.set_ticklabels(cls)
    
    plt.savefig(str(mfw) + str(column) + '_corpusDE_' + '.png')    
    
    plt.show()
    
    return error_dic

true_label = np.array(d.label)
d.drop('cosine', inplace=True, axis=1)
d.drop('label', inplace=True, axis=1)
d

cls = ['same', 'different']
error_list=[]
error_dic={}
for i, column in enumerate(d):
    print(column)
    cols = np.array(d[column])
    error_dic = normalized_cnf_matrix(cls, column, 2000) #ACHTUNG!alles irgendwie schöner machen, in eine Schleife
    error_dic['percentage'] = column
    error_list.append(error_dic)


errors = pd.DataFrame(error_list)
errors

minc = min(cos)
maxc = max(cos)
errors.name = ('min: {}, max: {}'.format(minc, maxc))

errors.name

errors.to_csv('errors_corpusDE')

# +
n = np.arange(0.01, 1.01, 0.01)
x = []
for perc in n:
    x.append(minc + cos_range*perc)
z = errors['fn']
y = errors['fp']
ax = plt.plot(x, z)
ax = plt.plot(x, y)
plt.xlabel("delta") 
plt.ylabel("true negatives") 

plt.show() 

# +
errors2 = errors.copy()
errors2.drop('tn', inplace=True, axis=1)
errors2.drop('tp', inplace=True, axis=1)
#errors2.drop('percentage', inplace=True, axis=1)

df = errors2.melt('percentage', var_name='errors',  value_name='vals')
df
# -

#sns.lineplot(data=errors2)
g = sns.lineplot(x="percentage", y="vals", hue='errors', data=df)

#berechne Schnittpunkt zwischen alpha und beta fehler
print((errors2.query('fn <= fp')).iloc[0].name)
(errors2.query('fp <= fn')).iloc[-1].name

df2=errors2.stack().reset_index()
df2.columns = ['Series','Event','Values']
print(df2)

plt.figure(figsize=(12,8))
ax = sns.pointplot(x='Series', y='Values', hue='Event',data=df2)
ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
plt.show()

# +
# visulisiere alpha und beta Fehler
# lege Delta-Kurven der verschiedenen Korpora übereinander, auch der Fehler evtl?
# verschiedene MFW-Werte austesten
# Streuung der Korpora
# -
# #### calculate mean length of chinese corpus

# +
import pandas as pd
import regex as re

def wordcounts_in_file(f_name):
    """

    :param f_name: filename of file to be analyzed
    :return: Counter of tokenized file
    """
    with open(f_name, encoding='utf-8') as f:
        # return Counter(remove_stopwords(f))
        return Counter(tokenize(f))

def tokenize(lines, token=re.compile(r'\p{L}+')):
    """

    :param lines (str): object to be tokenized, e.g. file
    :param token: pattern to tokenize lines
    :return: lowered and tokenized string
    """
    for line in lines:
        yield from map(str.lower, token.findall(line))


# +
import glob
from collections import Counter

path = 'dataset_fuer_mich/refcor-master/Chinese/*.txt'
length = []

for file in glob.glob(path):
    i=0
    count = wordcounts_in_file(file)
    for c in count:
        print(c)i = i + count[c]
    length.append(i)
length
# -

#i / 150
print(max(length), min(length))
sum(length)/75

explor = pd.read_csv('dataexploration_corpora.csv', encoding='utf-8')
explor.drop('max_min', axis=1, inplace=True)
explor.drop_duplicates(inplace=True)
explor

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table 
fig, ax = plt.subplots(figsize=(12, 3)) 
# no axes
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False)  
# no frame
ax.set_frame_on(False)  
# plot table
tab = table(ax, explor, loc='upper right')  
# set font manually
tab.auto_set_font_size(False)
tab.set_fontsize(8) 
# save the result
plt.savefig('dataexploration_corpora.png')

box = explor.copy()
box.drop('Mittlere Textlänge', axis=1, inplace=True)
box.drop('Textarten', axis=1, inplace=True)
box.drop('Anzahl Autoren', axis=1, inplace=True)
box.drop('Anzahl Texte', axis=1, inplace=True)
box.drop('Zeitraum', axis=1, inplace=True)
box.drop('balanciert', axis=1, inplace=True)
box

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.scatterplot(x='Korpus', y='Mittlere Textlänge', data=explor, color='.1', s=50)
sns.boxplot(x='Korpus', y='max_min', data=box, palette='binary')
plt.savefig("textlaenge_alle_korpora.png")


