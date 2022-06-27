# !/usr/bin/env python
# -*- coding:utf-8 -*-

import configparser

class BaseChecker:
    def __init__(self):
        pass
    
    def check(self):
        return True


class BinaryClfChecker(BaseChecker):
    def check(self):
        return True
    
    
class MultiClfChecker(BaseChecker):
    def check(self):
        return True
    
    
class ReggressionChecker(BaseChecker):
    def check(self):
        return true
    
    
class ClusterChecker(BaseChecker):
    def check(self):
        return true


class BaseParser:
    def __init__(self, parser, type_checker=BaseChecker()):
        self.parser = parser
        self.checker = type_checker
        
    def check(self):
        return self.checker.check()
    
    def parse(self, item_token):
        entries = []
        for item, value in self.parser.items(item_token):
            if value == 'true' and self.check():
                entries.append(item)
        return entries


#           model_type    metric session        model session       meta session    type checker
cfg_dict = {'binary':   ['binary_clf_metrics',  'clf_models',       'meta_models',  BinaryClfChecker()],
            'multi':    ['multi_clf_metrics',   'clf_models',       'meta_models',  MultiClfChecker()],
            'reg':      ['reg_metrics',         'reg_models',       None,           ReggressionChecker()],
            'cluster':  ['cluster_models',      'cluster_models',   None,           ClusterChecker()]}


class CfgParser:
    def __init__(self, cfg_file):
        self.__parser = configparser.ConfigParser()
        self.__parser.read(cfg_file)
        self.model_type = self.__parse_type()
        
        type_checker = cfg_dict[self.model_type][3]
        self.base_parser = BaseParser(parser=self.__parser, type_checker=type_checker)
        
    def __parse_type(self):
        return self.__parser.get('basic', 'model_type')

    def parse_metrics(self):
        metric_item = cfg_dict[self.model_type][0]
        return self.base_parser.parse(metric_item)
    
    def parse_models(self):
        model_item = cfg_dict[self.model_type][1]
        return self.base_parser.parse(model_item)
    
    def parse_meta_models(self):
        meta_item = cfg_dict[self.model_type][2]
        return self.base_parser.parse(meta_item)

    def parse_metrics_models(self):
        return self.parse_metrics(), self.parse_models()