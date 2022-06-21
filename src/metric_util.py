# !/usr/bin/env python
# -*- coding:utf-8 -*-

from registry import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from pyspark.mllib.evaluation import MulticlassMetrics

@register_obj
class SklMetric:
    def accuracy(self, y_true, y_pred, average=None, normalize=True, sample_weight=None):
        return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)

    def auc(self, x, y, reorder='deprecated'):
        return auc(x, y, reorder)

    def f1(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

    def precision(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

    def recall(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

    def roc_auc(self, y_true, y_score, average="macro", sample_weight=None, max_fpr=None):
        return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight, max_fpr=max_fpr)

    def confusion_matrix(self, y_true, y_pred, average=None, labels=None, sample_weight=None):
        return confusion_matrix(y_true, y_pred, labels, sample_weight=sample_weight)

@register_obj
class SpkMetric:
    def accuracy(self, prediction_and_labels):
        return MulticlassMetrics(prediction_and_labels)

    def auc(self, x, y, reorder='deprecated'):
        return None

    def f1(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return None

    def precision(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return None

    def recall(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
        return None

    def roc_auc(self, y_true, y_score, average="macro", sample_weight=None, max_fpr=None):
        return None

    def confusion_matrix(self, y_true, y_pred, average=None, labels=None, sample_weight=None):
        return None

class MetricUtil:
    def __init__(self, metric_type="SklMetric"):
        metric_obj = get_reg_obj(metric_type)
        self.metric_func_dict = {'accuracy': metric_obj.accuracy,
                                'f1': metric_obj.f1,
                                'precision': metric_obj.precision,
                                'recall': metric_obj.recall,
                                'roc_auc': metric_obj.roc_auc,
                                'confusion_matrix': metric_obj.confusion_matrix}
        
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
        metric_fn = self.metric_func_dict[metric]
        return metric_fn(y_true, y_pred, average=average)
    
    
    
        



