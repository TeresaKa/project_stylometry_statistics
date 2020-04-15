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
        data = pd.DataFrame(pd.read_hdf(self.file, index_col=0))
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
        base = base[base.cosine != 1.00]
        cosine = base.cosine.values.reshape(-1, 1)
        minx = min(cosine)
        cos = np.unique(cosine)

        # determine same and different authors in 1%-steps of cosine values
        cos_range = max(cos)-minx
        perc = np.arange(0.01, 1.01, 0.01)

        for p in perc:
            for c in cos:
                if c >= minx + cos_range*p:
                    base.loc[base.cosine == float(c), np.around(p, decimals=2)] = 'same'
                else:
                    base.loc[base.cosine == float(c), np.around(p, decimals=2)] = 'different'
        return base, cos


class ErrorCalculation:
    def __init__(self, predictions, cos, mfw, corpus):
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
        self.corpus = corpus
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

        plt.savefig(str(self.mfw) + '_' + str(self.corpus) + '_' + str(percentage) + '.png')
        plt.close(fig)
        #plt.show()

        #return ...  # ???

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
            cnf, error_dic = self.calculate_normalized_cnf_matrix(col_list, true_label)
            error_dic['percentage'] = column
            error_list.append(error_dic)
            #self.visualize_cnf_matrix(cnf, cls, column) # uncomment if confusion matrices for every percentage step are wanted


        errors = pd.DataFrame(error_list)
        errors.name = ('min: {}, max: {}'.format(self.min_cos, self.max_cos))

        errors.to_csv(str(self.mfw) + '_errors_' + str(self.corpus))
        return errors

    def extract_alpha_beta_errors(self):
        errors = self.extract_errors().copy()
        errors.drop('tn', inplace=True, axis=1)
        errors.drop('tp', inplace=True, axis=1)
        # errors2.drop('percentage', inplace=True, axis=1)
        return errors

    def visualise_alpha_beta_errors(self, errors):
        data = errors.melt('percentage', var_name='errors', value_name='vals')
        g = sns.lineplot(x="percentage", y="vals", hue='errors', data=data)
        plt.savefig(str(self.mfw) + '_error_' + str(self.corpus) + '.png')
        plt.show()
        return g

    def alpha_beta_intersection(self, errors):
        """ Find intersection point of alpha (false positive) and beta (false negative) error. """
        low = errors.copy()
        lower_margin = errors.copy().query('fp <= fn').iloc[-1].percentage
        low = low[low.percentage == lower_margin]
        low.loc[:, 'delta'] = min(cos) + (max(cos)-min(cos))*lower_margin

        high = errors.copy()
        upper_margin = errors.copy().query('fn <= fp').iloc[0].percentage
        high = high[high.percentage == upper_margin]
        high.loc[:, 'delta'] = min(cos) + (max(cos)-min(cos))*upper_margin

        best_values = pd.DataFrame(columns=['fp', 'fn', 'percentage', 'delta'])
        best_values = pd.concat([best_values, low])
        best_values = pd.concat([best_values, high])
        return best_values

    def put_all_together(self):
        errors = self.extract_alpha_beta_errors()
        self.visualise_alpha_beta_errors(errors)
        intersection = self.alpha_beta_intersection(errors)
        intersection['corpus'] = self.corpus
        intersection['mfw'] = self.mfw
        return intersection


path = 'results/Chinese/delta/*.h5'
prefix = 'results/Chinese/delta/'

best_values = pd.DataFrame(columns=['fp', 'fn', 'percentage', 'delta', 'corpus', 'mfw'])

for file in glob.glob(path):
    filename = file.replace(prefix, '').replace(file[-3:], '')
    mfw = filename.split('_')[0]
    corpus = filename.split('_')[2]
    print(filename, mfw, corpus)

    A = AuthorshipAttribution(file)
    data = A.reshape_dataframe()
    a, cos = A.delta_prediction()
    a.to_csv(str(mfw) + str(corpus) + '_attribution')
    print(a.to_string())

    E = ErrorCalculation(a, cos, mfw, corpus)
    intersection = E.put_all_together()
    best_values = pd.concat((best_values, intersection))
    #err = E.extract_errors()
    print(intersection)

best_values.to_csv(str(corpus) + '_best_cutoff_values.csv')