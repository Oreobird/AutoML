# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from sklearn import model_selection
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV

import metric_util as mtutil
import model_util as mdutil

class BaseAutoML:
    def __init__(self, model_util=mdutil.ModelUtil(), model_save_path='./'):
        self.model_util = model_util
        self.model_save_path = model_save_path
        self.selected_models = []
        self.best_model = None
        
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
    def __print_best_score(self, gs, param_test):
        print("Best score: %0.3f" % gs.best_score_)
        print("Best parameters set:")
    
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(param_test.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
            
    def cvselect_model(self, x_train, y_train, metric_list, model_name_list, K=2, cv=5):
        select_models = []
        for metric in metric_list:
            scores = {}
            print("===>[%s]" % metric)
            for model_name in model_name_list:
                model = self.model_util.get_model(model_name)
                score = model_selection.cross_val_score(model, x_train, y_train, cv=cv, scoring=metric)
                scores[model_name] = score.mean()
                print("    [%s] Accuracy: %0.2f (+/- %0.2f)" % (model_name, score.mean(), score.std()))
            sorted_scores = sorted(scores.items(), key=lambda d:d[1], reverse = True)
            selected_model_name = sorted_scores[:K]
            
            for i in range(len(selected_model_name)):
                if not selected_model_name[i][0] in select_models:
                    select_models.append(selected_model_name[i][0])
        
        self.selected_models = select_models[:K]
        print("Selected models:{}".format(self.selected_models))
        
        return self.selected_models[:K]


    def model_param_tune(self, x_train, y_train, model, model_params, scoring='accuracy', cv=5):
        gs = GridSearchCV(estimator=model, param_grid=model_params, scoring=scoring, cv=cv)
        gs.fit(x_train, y_train)
        best_model = gs.best_estimator_
        self.__print_best_score(gs, model_params)
        return best_model


    def tune_models(self, x_train, y_train, model_names=None, save=True):
        best_models = {}
        for model_name in model_names:
            model = self.model_util.get_model(model_name)
            param_set = self.model_util.get_param_set(model_name)
            best_model = self.model_param_tune(x_train, y_train, model, param_set)
            if save:
                self.model_util.save_model(best_model, os.path.join(self.model_save_path, model_name + '_model.pkl'))
            best_models[model_name] = best_model
        return best_models


    def get_basic_models(self, models_name_list):
        classifiers = []
        for model_name in models_name_list:
            model_path = os.path.join(self.model_save_path, model_name + '_model.pkl')
            if not os.path.exists(model_path):
                basic_model = self.model_util.get_model(model_name)
            else:
                basic_model = self.model_util.load_model(model_path)
                print("Load model:" + model_name)
            classifiers.append((model_name, basic_model))
        return classifiers

    def get_models_params(self, models_name_list):
        params = {}
        for model_name in models_name_list:
            model_param = self.model_util.get_param_set(model_name)
            for k, v in model_param.items():
                params[model_name + '__' + k] = v
        return params
    
    def stacking(self, x_train, y_train, meta_clf_name=None, save=True, name='stack_model.pkl'):
        if meta_clf_name is None:
            return
        meta_clf = self.model_util.get_model(meta_clf_name)
        stack_clf = StackingClassifier(estimators=self.get_basic_models(self.selected_models),
                                       final_estimator=meta_clf)

        models_name_list = self.selected_models
        params = self.get_models_params(models_name_list)

        self.best_model = self.model_param_tune(x_train, y_train, stack_clf, params)
        if save:
            self.model_util.save_model(self.best_model, os.path.join(self.model_save_path, name))
        return self.best_model

    def save_model(self, model_path=None):
        self.model_util.save_model(self.best_model, model_path)
        
    def load_model(self, model_path):
        return self.model_util.load_model(model_path)
    

class AutoML(BaseAutoML):
    def train(self, x_train, y_train, metric_list, model_name_list, meta_model_name, model_save_name, K=2):
        selected_models = super().cvselect_model(x_train, y_train, metric_list, model_name_list, K=K)
        super().tune_models(x_train, y_train, selected_models)
        return super().stacking(x_train, y_train, meta_model_name, name=model_save_name)
    
    def validate(self, model, x_val, y_val, metrics):
        predict_y = model.predict(x_val)
        mt = mtutil.MetricUtil()
        for metric in metrics:
            print('{}:{}'.format(metric, mt.metric_score(y_val, predict_y, metric)))
        return predict_y
    
    def predict(self, model, x_test):
        predict_y = model.predict(x_test)
        return predict_y

        

