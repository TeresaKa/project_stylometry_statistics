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
        """
        Prepare data calculated in 'delta.py' for further use. Assign authorship using Cosine Delta values.
        Sets decision boundaries in 1%-steps added to minimum delta value.
        :param file: pd.Dataframe with Delta values
        """
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
        cosine = base['cosine'].values.reshape(-1, 1)
        minx = min(cosine)
        cos = np.unique(cosine)

        # determine same and different authors in 1%-steps of cosine values
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
    def __init__(self, predictions, cos, mfw):
        """
        Calculates true positive, true negative, false positive and false negative values.
        :param predictions: Predicted authorship attributions as pd.Dataframe
        :param cos: List of unique Cosinus Delta values in 'predictions'
        """
        self.predictions = predictions   # use delta_prediction
        self.cos = cos
        self.min_cos = min(self.cos)
        self.max_cos = max(self.cos)
        self.mfw = mfw
# ### calculate and visualize alpha and beta errors

    def calculate_normalized_cnf_matrix(self, true_label, cols):
        """ Create Confusion Matrix and extract true negative, true positive, false negative, false positive values """
        cnf = confusion_matrix(true_label, cols, normalize='all')
        print(cnf)
        tn, fp, fn, tp = cnf.ravel()

        error_dic = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

        return cnf, error_dic

    def visualize_cnf_matrix(self, cnf, cls, percentage):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(cnf, annot=True, cmap=sns.color_palette("Blues"), ax = ax)

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(cls)
        ax.yaxis.set_ticklabels(cls)

        plt.savefig(str(self.mfw) + '_' + str(percentage) + '_' + str(corpus) + '.png')
        plt.close(fig)
        #plt.show()

        return ...  # ???

    def extract_errors(self):
        """ Create pd.Dataframe with true negative, true positive, false negative, false positive values
        for every percentage-step for decision boundary """
        true_label = np.array(self.predictions.label)
        self.predictions.drop('cosine', inplace=True, axis=1)
        self.predictions.drop('label', inplace=True, axis=1)
        cls = ['same', 'different']
        error_list = []
        for i, column in enumerate(self.predictions):
            print(column)
            col_list = np.array(self.predictions[column])
            cnf, error_dic = self.calculate_normalized_cnf_matrix(col_list, true_label) #ACHTUNG!alles irgendwie schöner machen, in eine Schleife
            error_dic['percentage'] = column
            error_list.append(error_dic)
            #self.visualize_cnf_matrix(cnf, cls, column) # uncomment if confusion matrices for every percentage step are wanted


        errors = pd.DataFrame(error_list)
        errors.name = ('min: {}, max: {}'.format(self.min_cos, self.max_cos))

        errors.to_csv('errors_' + str(self.mfw) + '_' + str(corpus))
        return errors

    def extract_alpha_beta_errors(self):
        errors = self.extract_errors(self.mfw).copy()
        errors.drop('tn', inplace=True, axis=1)
        errors.drop('tp', inplace=True, axis=1)
        # errors2.drop('percentage', inplace=True, axis=1)

        return errors

    def visualise_alpha_beta_errors(self):
        data = self.extract_alpha_beta_errors(self.mfw).melt('percentage', var_name='errors', value_name='vals')
        g = sns.lineplot(x="percentage", y="vals", hue='errors', data=data)
        plt.savefig('error_' + str(corpus) + '.png')
        plt.show()
        return g

    def alpha_beta_intersection(self):
        """ Find intersection point of alpha (false positive) and beta (false negative) error. """
        errors = self.extract_alpha_beta_errors(self.mfw)
        lower_margin = errors.query('fp <= fn').iloc[-1].percentage
        fp_low = errors.query('fp <= fn').iloc[-1].fp
        fn_low = errors.query('fp <= fn').iloc[-1].fn
        upper_margin = errors.query('fn <= fp').iloc[0].percentage
        fp_up = errors.query('fp <= fn').iloc[0].fp
        fn_up = errors.query('fp <= fn').iloc[0].fn
        intersection = {'lower_margin':lower_margin, 'upper_margin':upper_margin, 'fp_low':fp_low, 'fp_up':fp_up,
                        'fn_low': fn_low, 'fn_up': fn_up}
        return intersection

path = 'results/Chinese/delta/*.h5'
prefix = 'results/Chinese/delta/'
#corpus = 'piperDE'

for file in glob.glob(path):
    filename = file.replace(prefix, '').replace(file[-3:], '')
    mfw = filename.split('_')[0]
    corpus = filename.split('_')[2]
    print(filename, mfw, corpus)

    A = AuthorshipAttribution(file)
    a, cos = A.delta_prediction()
    print(a.to_string())

    E = ErrorCalculation(a, cos, mfw)
    intersection = E.alpha_beta_intersection()
    intersection['corpus'] = corpus
    intersection['mfw'] = mfw
    intersection['min_cos'] = min(cos)
    err = E.extract_errors()
    print(err)

    blub = [corpus, mfw, x, y]
    import csv
    with open(r'cutoffs_entire_data.csv', 'a') as f:
        cutoff = csv.writer(f)
        cutoff.writerow(blub)


# lege Delta-Kurven der verschiedenen Korpora übereinander, auch der Fehler evtl?
# Streuung der Korpora
# bei Visualisierungs-Notebook: cutoff-csv importieren, Prozentwerte vis (?), cnf Matrizen daraus ableiten
