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


