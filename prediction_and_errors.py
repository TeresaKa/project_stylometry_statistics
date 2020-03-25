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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import glob

class AuthorshipAttribution:
    def __init__(self, file):
        self.file = file
# ### load calculated delta measures

    def load_deltas(self):
        data = pd.DataFrame(pd.read_csv(self.file, index_col=0))
        data = data[data.cosine != 0.00]
        return data

    def reshape_dataframe(self):
        d = self.load_deltas().copy()
        d.drop('manhattan', inplace=True, axis=1)
        d.drop('euclidean', inplace=True, axis=1)
        d.drop('author', inplace=True, axis=1)
        return d

    def delta_prediction(self):
        base = self.reshape_dataframe()
        #was passiert hier:
        cosine = base['cosine'].values.reshape(-1, 1)  #np.unique machen?in Schleife unten werden trotzdem alle Zeilen erfasst damit
        minx = min(cosine)
        cos = np.unique(cosine)

        # ### determine same and different authors in 1%-steps of cosine values
        cos_range = max(cos)-min(cos)
        perc = np.arange(0.01, 1.01, 0.01)

        for p in perc:
            for c in cos:
                if c <= minx + cos_range*p:
                    base.loc[base.cosine == float(c), np.around(p, decimals=2)] = 'same'
                else:
                    base.loc[base.cosine == float(c), np.around(p, decimals=2)] = 'different'
        return base, cos


class ErrorCalculation:
    def __init__(self, predictions, cos):
        self.predictions = predictions   # use delta_prediction
        self.cos = cos
        self.min_cos = min(self.cos)
        self.max_cos = max(self.cos)
# ### calculate and visualize alpha and beta errors

    def normalized_cnf_matrix(self, cls, column, cols, true_label):
       # plt.ioff()   #disable displaying plots

        cnf = confusion_matrix(true_label, cols, normalize='all')
        fig, ax = plt.subplots(figsize=(5,5))
        #plt.figure(figsize=(5,5))
        sns.heatmap(cnf, annot=True, cmap=sns.color_palette("Blues"), ax = ax)


        tn, fp, fn, tp = cnf.ravel()

        error_dic = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(cls)
        ax.yaxis.set_ticklabels(cls)

        plt.savefig(str(mfw) + '_' + str(column) + '_' + str(corpus) + '.png')
        plt.close(fig)
        #plt.show()

        return error_dic

    def calculate_errors(self):
        true_label = np.array(self.predictions.label)
        self.predictions.drop('cosine', inplace=True, axis=1)
        self.predictions.drop('label', inplace=True, axis=1)
        cls = ['same', 'different']
        error_list = []
        # error_dic = {}
        for i, column in enumerate(self.predictions):
            print(column)
            col_list = np.array(self.predictions[column])
            error_dic = self.normalized_cnf_matrix(cls, column, col_list, true_label) #ACHTUNG!alles irgendwie schöner machen, in eine Schleife
            error_dic['percentage'] = column
            error_list.append(error_dic)

        errors = pd.DataFrame(error_list)
        errors.name = ('min: {}, max: {}'.format(self.min_cos, self.max_cos))

        errors.to_csv('errors_' + str(corpus))
        return errors


    def extract_alpha_beta_errors(self):
        errors = self.calculate_errors().copy()
        errors.drop('tn', inplace=True, axis=1)
        errors.drop('tp', inplace=True, axis=1)
        # errors2.drop('percentage', inplace=True, axis=1)

        return errors


    def visualise_errors(self):
        data = self.extract_alpha_beta_errors().melt('percentage', var_name='errors', value_name='vals')
        g = sns.lineplot(x="percentage", y="vals", hue='errors', data=data)
        plt.savefig('error_' + str(corpus) + '.png')
        plt.show()
        return g

    def alpha_beta_intersection(self):
        #berechne Schnittpunkt zwischen alpha und beta fehler
        errors = self.extract_alpha_beta_errors()
        # lower_margin = errors.query('fp <= fn').iloc[-1].name
        # upper_margin = errors.query('fn <= fp').iloc[0].name
        lower_margin = errors.query('fp <= fn').iloc[-1].percentage
        upper_margin = errors.query('fn <= fp').iloc[0].percentage
        #save to file with cos max and min values, corpus etc.
        return lower_margin, upper_margin


path = 'results/piperDE/delta/*.h5'
prefix = 'results/piperDE/delta/'
#corpus = 'piperDE'

for file in glob.glob(path):
    filename = file.replace(prefix, '').replace(file[-3:], '')
    mfw = filename.split('_')[0]
    corpus = filename.split('_')[2]
    print(filename, mfw, corpus)

    A = AuthorshipAttribution(file)
    a, cos = A.delta_prediction()
    print(a.to_string())

    E = ErrorCalculation(a, cos)
    #x, y = E.alpha_beta_intersection()
    err = E.visualise_errors()
    print(err)


# lege Delta-Kurven der verschiedenen Korpora übereinander, auch der Fehler evtl?
# Streuung der Korpora
