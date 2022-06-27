# !/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
import config_parser
import automl_base
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

CFG_FILE_PATH = '../config/'
DATA_FILE_PATH = '../data/'
MODEL_FILE_PATH = '../model/'

def cfg_parser_test():
    cfg = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'config_template.ini'))
    
    metrics = cfg.parse_metrics()
    models = cfg.parse_models()
    meta_model = cfg.parse_meta_models()
    
    print(metrics)
    print(models)
    print(meta_model)
    
    
def multi_model_train_test():
    iris = datasets.load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data[:, 1:3], iris.target, test_size=0.3, random_state=42)
    
    cfg = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'multi_config.ini'))

    metric_list, model_label_list = cfg.parse_metrics_models()
    meta_model_label = cfg.parse_meta_models()
   
    automl = automl_base.AutoML(model_save_path=os.path.join(MODEL_FILE_PATH, 'iris_models/'))
    model = automl.train(X_train, Y_train, metric_list, model_label_list, meta_model_label[0], model_save_name='iris_model.pkl', K=3)

def multi_model_predict_test():
    iris = datasets.load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data[:, 1:3], iris.target, test_size=0.3, random_state=42)
    
    automl = automl_base.AutoML(model_save_path=os.path.join(MODEL_FILE_PATH, 'iris_models/'))
    cfg = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'multi_config.ini'))
    metric_list = cfg.parse_metrics()
    
    model = automl.load_model(os.path.join(MODEL_FILE_PATH, 'iris_models/iris_model.pkl'))
    val_y = automl.validate(model, X_test, Y_test, metric_list)
    pred_y = automl.predict(model, X_test)

def binary_model_data_prepare():
    path_train = os.path.join(DATA_FILE_PATH, 'train.csv')
    path_test = os.path.join(DATA_FILE_PATH, 'test.csv')
    
    def data_explor(df_raw):
        print(df_raw.head())
        
        print(df_raw.describe())
        print(df_raw.info())
        
        plot = sns.catplot(x="Embarked", y="Fare", hue="Sex", data=df_raw, height=6, kind="bar", palette="muted")
        plot.set_ylabels("Pclass")
        plt.show()
        
        embarked_null = df_raw[df_raw['Embarked'].isnull()]
        print(embarked_null)
        
        df_raw.drop(['PassengerId'], 1).hist(bins=50, figsize=(20, 15))
        plt.show()
    
    def preprocess_data(df):
        process_df = df

        # 1.Deal with missing values
        
        process_df['Embarked'].fillna('C', inplace=True)
        
        # replace missing age by the mean age of passengers who belong to the same group of class/sex/family
        process_df['Age'] = process_df.groupby(['Pclass', 'Sex', 'Parch', 'SibSp'])['Age'].transform(
            lambda x: x.fillna(x.mean()))
        process_df['Age'] = process_df.groupby(['Pclass', 'Sex', 'Parch'])['Age'].transform(
            lambda x: x.fillna(x.mean()))
        process_df['Age'] = process_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
        
        # replace the only missing fare value for test dataset and the missing values of the cabin column
        process_df['Fare'] = process_df['Fare'].interpolate()
        process_df['Cabin'].fillna('U', inplace=True)

        # 2. Feature engineeing on columns
        # Create a title column from name column
        process_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train_df_raw['Name']),
                                        index=train_df_raw.index)
        process_df['Title'] = process_df['Title'].replace(
            ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            'Rare')
        process_df['Title'] = process_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        process_df['Title'] = process_df['Title'].replace('Mme', 'Mrs')
        process_df['Title'] = process_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
        
        # Filling Age missing values with mean age of passengers who have the same title
        process_df['Age'] = process_df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))
        
        # print(process_df['Age'])
        
        # Transform categorical variables to numeric variables
        process_df['Sex'] = process_df['Sex'].map({'male': 0, 'female': 1})
        process_df['Embarked'] = process_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Create a Family size, Is alone, child and mother columns
        process_df['FamillySize'] = process_df['SibSp'] + process_df['Parch'] + 1
        process_df['FamillySize'][process_df['FamillySize'].between(1, 5, inclusive=False)] = 2
        process_df['FamillySize'][process_df['FamillySize'] > 5] = 3
        process_df['IsAlone'] = np.where(process_df['FamillySize'] != 1, 0, 1)
        process_df['IsChild'] = process_df['Age'] < 18
        process_df['IsChild'] = process_df['IsChild'].astype(int)
        
        # Modification of cabin to keep only the letter contained corresponding to the deck of the boat
        process_df['Cabin'] = process_df['Cabin'].str[:1]
        process_df['Cabin'] = process_df['Cabin'].map(
            {cabin: p for p, cabin in enumerate(set(cab for cab in process_df['Cabin']))})
        
        # Create a ticket survivor column
        process_df['TicketSurvivor'] = pd.Series(0, index=process_df.index)
        tickets = process_df['Ticket'].value_counts().to_dict()
        for t, occ in tickets.items():
            if occ != 1:
                table = train_df_raw['Survived'][train_df_raw['Ticket'] == t]
                if sum(table) != 0:
                    process_df['TicketSurvivor'][process_df['Ticket'] == t] = 1
        
        # drop not useful anymore
        process_df = process_df.drop(['Name', 'Ticket', 'PassengerId'], 1)
        
        return process_df
    
    train_df_raw = pd.read_csv(path_train)
    test_df_raw = pd.read_csv(path_test)
    
    # data_explor(df_raw)
    
    train_df = train_df_raw.copy()
    
    X = train_df.drop(['Survived'], 1)
    Y = train_df['Survived']
    
    X = preprocess_data(X)
    # print(X)
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)

    test_df = test_df_raw.copy()
    X_test = preprocess_data(test_df)
    X_test = pd.DataFrame(sc.fit_transform(X_test.values), index=X_test.index, columns=X_test.columns)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_val, Y_train, Y_val, X_test
    
def binary_model_train_test():
    cfg = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'binary_config.ini'))
    metric_list, model_label_list = cfg.parse_metrics_models()
    meta_model_label = cfg.parse_meta_models()
    if len(meta_model_label) > 0:
        meta_model_label = meta_model_label[0]
    else:
        meta_model_label = None
    automl = automl_base.AutoML(model_save_path=os.path.join(MODEL_FILE_PATH, 'titanic_models/'))
    
    X_train, X_val, Y_train, Y_val, X_test = binary_model_data_prepare()
    model = automl.train(X_train, Y_train, metric_list, model_label_list, meta_model_label, 'titanic_model.pkl', K=3)
    
def binary_model_predict_test():
    cfg = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'binary_config.ini'))
    metric_list, model_label_list = cfg.parse_metrics_models()
   
    automl = automl_base.AutoML(model_save_path=os.path.join(MODEL_FILE_PATH, 'titanic_models/'))
    model = automl.load_model(os.path.join(MODEL_FILE_PATH, 'titanic_models/titanic_model.pkl'))
    
    X_train, X_val, Y_train, Y_val, X_test = binary_model_data_prepare()
    val_y = automl.validate(model, X_val, Y_val, metric_list)
    pred_y = automl.predict(model, X_test)
    
def run_test_case():
    cfg_parser_test()
    binary_model_train_test()
    binary_model_predict_test()
    multi_model_train_test()
    multi_model_predict_test()
    
if __name__ == '__main__':
    run_test_case()