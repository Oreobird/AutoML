# !/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2019/3/15
# author: zgs

import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# from pyspark.mllib.evaluation import MulticlassMetrics

class MetricUtil:
    def __init__(self):
        self.metric_dict = {'accuracy': self.sk_accuracy,
                            'f1': self.sk_f1,
                            'precision': self.sk_precision,
                            'recall': self.sk_recall,
                            'roc_auc': self.sk_roc_auc,
                            'confusion_matrix': self.sk_confusion_matrix}
        
    def __parse_average(self, metric_str=None):
        average = None
        if metric_str and '_' in metric_str and not 'roc' in metric_str:
            average = metric_str.split('_')[-1]
        return average
    
    def __parse_metric(self, metric_str=None):
        metric = metric_str
        if metric_str and '_' in metric_str and not 'roc' in metric_str:
            metric = metric_str.split('_')[0]
        return metric
        
    def metric_score(self, y_true, y_pred, metric_str='accuracy'):
        metric = self.__parse_metric(metric_str)
        average = self.__parse_average(metric_str)
        # print("metric:{}, average:{}".format(metric, average))
        metric_fn = self.metric_dict[metric]
        return metric_fn(y_true, y_pred, average=average)
        
    def sk_accuracy(self, y_true, y_pred, average=None, normalize=True, sample_weight=None):
        return accuracy_score(y_true, y_pred, normalize, sample_weight)
    
    def sk_auc(self, x, y, reorder='deprecated'):
        return auc(x, y, reorder)
    
    def sk_f1(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return f1_score(y_true, y_pred, labels, pos_label, average, sample_weight)
    
    def sk_precision(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return precision_score(y_true, y_pred, labels, pos_label, average, sample_weight)
    
    def sk_recall(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return recall_score(y_true, y_pred, labels, pos_label, average, sample_weight)
    
    def sk_roc_auc(self, y_true, y_score, average="macro", sample_weight=None, max_fpr=None):
        return roc_auc_score(y_true, y_score, average, sample_weight, max_fpr)
    
    def sk_confusion_matrix(self, y_true, y_pred, average=None, labels=None, sample_weight=None):
        return confusion_matrix(y_true, y_pred, labels, sample_weight)
    
    def sp_accuracy(self, prediction_and_labels):
        return MulticlassMetrics(prediction_and_labels)
    
    
    
    
        



