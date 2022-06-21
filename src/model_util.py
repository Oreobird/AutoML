# !/usr/bin/env python
# -*- coding:utf-8 -*-

from registry import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm
import xgboost

from sklearn.cluster import KMeans
import joblib

# from pyspark.mllib.classification import LogisticRegressionWithSGD
# from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# from pyspark.mllib.classification import SVMWithSGD
# from pyspark.mllib.classification import SVMModel
# from pyspark.mllib.classification import NaiveBayes
# from pyspark.mllib.classification import NaiveBayesModel
#
# from pyspark.mllib.clustering import KMeans
# from pyspark.mllib.clustering import KMeansModel
# from pyspark.mllib.clustering import GaussianMixture
# from pyspark.mllib.clustering import GaussianMixtureModel
#
#
# from pyspark.mllib.tree import DecisionTree
# from pyspark.mllib.tree import DecisionTreeModel
# from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.tree import RandomForestModel
# from pyspark.mllib.tree import GradientBoostedTrees
# from pyspark.mllib.tree import GradientBoostedTreesModel
#
# from pyspark.ml.classification import LogisticRegression

class ModelBase:
    def __init__(self):
        self.__name = None
        self.__obj = None
        self.__param_space = {}

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def obj(self):
        return self.__obj

    @obj.setter
    def obj(self, obj):
        self.__obj = obj

    @property
    def param_space(self):
        return self.__param_space

    @param_space.setter
    def param_space(self, param_dict):
        self.__param_space = param_dict

@register_obj('lr')
class LRModel(ModelBase):
    def __init__(self):
        ModelBase.obj = LogisticRegression(multi_class='auto', solver='lbfgs', penalty='l2', verbose=0)
        ModelBase.param_space = {'C': [x / 10.0 for x in range(1, 50, 5)],
                                  'max_iter': [50, 100, 500],
                                  'multi_class': ['auto'],
                                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                  'class_weight': ['balanced']}

@register_obj('decisiontree')
class DecisionTreeClsModel(ModelBase):
    def __init__(self):
        ModelBase.obj = DecisionTreeClassifier()
        ModelBase.param_space = {'max_depth': range(2, 8, 1),
                               'min_samples_split': [x / 10.0 for x in range(1, 5, 2)],
                               'min_samples_leaf': [x / 10.0 for x in range(1, 5, 2)],
                               'min_weight_fraction_leaf': [x / 10.0 for x in range(0, 5, 2)],
                               'min_impurity_decrease': [x / 10.0 for x in range(1, 20, 5)]}

@register_obj('gaussiannb')
class GaussianNBModel(ModelBase):
    def __init__(self):
        ModelBase.obj = GaussianNB()
        ModelBase.param_space = {}

@register_obj('svm')
class SVMModel(ModelBase):
    def __init__(self):
        ModelBase.obj = SVC(gamma='scale')
        ModelBase.param_space = {'C': [x / 10.0 for x in range(1, 20, 5)]}

@register_obj('knn')
class KNNModel(ModelBase):
    def __init__(self):
        ModelBase.obj = KNeighborsClassifier()
        ModelBase.param_space = {'n_neighbors': range(2, 10, 2),
                               'leaf_size': range(10, 50, 5)}

@register_obj('randomforest')
class RandomForestClsModel(ModelBase):
    def __init__(self):
        ModelBase.obj = RandomForestClassifier(n_estimators=10, random_state=1)
        ModelBase.param_space = {'n_estimators': range(10, 200, 10),
                                'max_depth': range(2, 8, 1),
                                'min_samples_split': [x / 10.0 for x in range(1, 5, 2)],
                                'min_samples_leaf': [x / 10.0 for x in range(1, 5, 2)],
                                'min_weight_fraction_leaf': [x / 10.0 for x in range(1, 5, 2)],
                                'min_impurity_decrease': [x / 10.0 for x in range(1, 20, 5)],
                                'bootstrap': [True, False],
                                'oob_score': [True, False]}

@register_obj('xgboost')
class XGBClsModel(ModelBase):
    def __init__(self):
        ModelBase.obj = xgboost.XGBClassifier(eval_metric=['logloss', 'auc'])
        ModelBase.param_space = {'max_depth': range(4, 12, 2),
                                'learning_rate': [x / 10.0 for x in range(1, 10, 2)],
                                'n_estimators': range(50, 100, 20)}

@register_obj('lgbm')
class LGBMClsModel(ModelBase):
    def __init__(self):
        ModelBase.obj = lightgbm.LGBMClassifier()
        ModelBase.param_space = {'num_leaves': range(2, 10, 2),
                                 'max_depth': range(2, 6, 2)}

@register_obj('k-means')
class KMeansModel(ModelBase):
    def __init__(self):
        ModelBase.obj = KMeans()
        ModelBase.param_space = {'n_clusters': range(2, 20, 2),
                                   'n_init': range(4, 20, 2),
                                   'max_iter': range(200, 400, 100)}

class ModelUtil:
    def get_model(self, name='randomforest'):
        return get_reg_obj(name).obj
    
    def get_param_set(self, name='randomforest'):
        return get_reg_obj(name).param_space
    
    def save_model(self, model, file):
        joblib.dump(model, file)

    def load_model(self, file):
        return joblib.load(file)


