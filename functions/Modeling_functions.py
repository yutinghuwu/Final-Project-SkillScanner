#!/usr/bin/env python
# coding: utf-8

from linkedin_api import Linkedin
import pandas as pd
import pickle
import regex as re
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import plot_roc_curve


def create_category (data, title_column, category_column):
    '''
    Given a dataframe a column create a new column with a job category according if the job title have the keywords
    
    Parameters
    ----------
    data : dataframe
    title_column : string, column name in the dataframe 
    category_column : string, new column to be created as an output
    
    Output
    ----------
    The categories that will be use to map the first column given are:
    'Data Analyst'
    'Data Engineer'
    'Data Scientist'
    The categories keywords must be present in teh title as strings in lowercase or uppercase.
    If there are missing values the function will return a null value for that row
    '''
    
    data[title_column].fillna('Unknown', inplace = True)
    
    for i in data[data[title_column].str.contains('Analy', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Analyst'
        
    for i in data[data[title_column].str.contains('Intelligen',na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Analyst'

    for i in data[data[title_column].str.contains('Engineer', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Engineer'
        
    for i in data[data[title_column].str.contains('Data\s*\w*\s*\w*\s*Architect',na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Engineer'        
        
    for i in data[data[title_column].str.contains('Scien', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    for i in data[data[title_column].str.contains('ML', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    for i in data[data[title_column].str.contains('Machine\s*\w*\s*\w*\s*Learning', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    return data

def crea_cat (x, list_cat, str_cat):
    '''
    Given a text and  list of string, find if the strings are present in the text and return a category 
    
    Parameters
    ----------
    x : string
    list_cat : list of strings 
    str_cat : string, new label to be return as an output
    
    Output
    ----------
    If the funtion find any element of the list from the second parameter present in the first string
    will return a string given as the third parameter, in form of a category to be assign.
    '''
    for skill in list_cat:
        if skill in x:
            return str_cat      
        
        
def find_only_whole_word(skill, description):
    '''
    Create a raw string with word boundaries from the user's input_string
    
    Parameters
    ----------
    skill : string
    description: string
    
    The first string must be a one caracther only string to be found in the second string representing a larger text.
    
    Output
    ----------
    The funtion return a 0 if no match was found or a 1 if there was a match
    
    '''
    raw_search_string = r"\b" + skill + r"\b"
    match_output = re.search(raw_search_string, description,
                          flags=re.IGNORECASE)
    no_match_was_found = ( match_output is None )
    if no_match_was_found:
        return 0
    else:
        return 1

def metrics_df(test_real_class, test_predicted_class, train_real_class, train_predicted_class, model_name):
    '''
    Given the real labels and the result of the predition from a model, get a dataframe with the comparation of
    the metrics of accuracy, precision, recall and f1 for train and test preditions.
    
    Parameters
    ----------
    test_real_class: Series, real labels from the classification dataframe 
    test_predicted_class: numpy.ndarray, result of the prediction from the model
    train_real_class: Series, real labels from the classification dataframe
    train_predicted_class: numpy.ndarray, result of the prediction from the model
    model_name: string

    Output
    ----------
    Dataframe with the scores as values and each set (train and test) as rows. 
    The Dataframe will identify each set and put a name to the model used according to the last string given as a parameter
    '''
    
    # para el test
    accuracy_test = accuracy_score(test_real_class, test_predicted_class)
    precision_test = precision_score(test_real_class, test_predicted_class, average='micro' )
    recall_test = recall_score(test_real_class, test_predicted_class, average='micro' )
    f1_test = f1_score(test_real_class, test_predicted_class, average='micro' )

    # para el train
    accuracy_train = accuracy_score(train_real_class, train_predicted_class)
    precision_train = precision_score(train_real_class, train_predicted_class, average='micro' )
    recall_train = recall_score(train_real_class, train_predicted_class, average='micro' )
    f1_train = f1_score(train_real_class, train_predicted_class, average='micro' )
   
        
    df = pd.DataFrame({"accuracy": [accuracy_test, accuracy_train], 
                       "precision": [precision_test, precision_train],
                       "recall": [recall_test, recall_train], 
                       "f1": [f1_test, f1_train],
                       "set": ["test", "train"]} 
                        )
    
    df["modelo"] = model_name
    return df


def metrics_concat(test_real_class, test_predicted_class, train_real_class, train_predicted_class, model_name, df_all_results):
    results =  metrics_df(test_real_class, test_predicted_class, train_real_class, train_predicted_class, model_name)
    '''
    Given the real labels and the result of the predition from a model, create a new dataframe with the comparation of
    the metrics of accuracy, precision, recall and f1 for train and test preditions and concat to a previous dataframe given.
    
    Parameters
    ----------
    test_real_class: Series, real labels from the classification dataframe 
    test_predicted_class: numpy.ndarray, result of the prediction from the model
    train_real_class: Series, real labels from the classification dataframe
    train_predicted_class: numpy.ndarray, result of the prediction from the model
    model_name: string
    df_all_results: dataframe, with same row and columns structure

    Output
    ----------
    Dataframe with the scores as values and each set (train and test) as rows. 
    The new scores given in this funtion will appear as the last rows of the previous dataframe.
    '''
    df_all_results = pd.concat([df_all_results, results], axis = 0)
    return df_all_results

