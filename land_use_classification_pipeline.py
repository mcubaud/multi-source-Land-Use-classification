#%% -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:25:22 2022

@author: MCubaud
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

#scikit-learn
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

#Imbalanced learn
from imblearn.over_sampling import SMOTENC, SMOTEN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import datetime
import time

import skops.io as sio

#functions defined in other files
from definition_of_sources import Sources
from plot_functions import recall_matrix_with_sep_diag, precision_matrix_with_sep_diag, F1_recall_precision_barh


def per_class_metrics(conf_mat):
    """Compute the per class metrics from the confusion matrix."""
    diag = np.diag(conf_mat)
    row_sums = np.sum(conf_mat, axis=1)
    col_sums = np.sum(conf_mat, axis=0)
    #If a class is present but not predicted precision=0
    # if it is not present and not predicted precision=nan
    precisions = np.where((col_sums == 0) & (row_sums > 0), 0, diag / col_sums)
    #If a class is predicted but not present recall=0
    # if it is not present and not predicted recall=nan
    recalls = np.where((row_sums == 0) & (col_sums > 0), 0, diag / row_sums)
    F1 = 2 * recalls*precisions/(recalls+precisions)
    F1 = np.where((precisions == 0) | (recalls == 0), 0, F1)
    return recalls, precisions, F1

def divide_max_by_2(y):
    """Downscaling strategy: the majority class is being set to half its current size."""
    uniques, counts = np.unique(y, return_counts=True)
    counts[np.argmax(counts)] /= 2
    return {uniques[i]:counts[i] for i in range(len(uniques))}

def if_less_than_1000_set_1000(y):
    """Upscaling strategy: all the classes with less than 1000 samples are upsampled to 1000 samples."""
    value = 1000
    uniques, counts = np.unique(y, return_counts=True)
    counts[counts<value] = value
    return {uniques[i]:counts[i] for i in range(len(uniques))}

def train_test(
        clf,
        X_train,
        y_train,
        X_test,
        y_test
        ):
    """Train the classifier on the training set and apply it on the test set."""
    
    #TRAIN
    print("train")
    t0 = time.time()
    clf.fit(X_train, y_train)
    # clf.fit(X_train, y_train, clf__eval_set=[(X_eval, y_eval)])
    train_time = time.time() - t0
    y_train_pred = clf.predict(X_train)
    
    #TEST
    print("test")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - t0
    
    return y_train_pred, train_time, y_pred, test_time, clf
    

def train_test_and_save_result(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        X_geom,
        clf_name,
        combination,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment
        ):
    """Train the classifier on the training set, apply it on the test set and save the results.

    Parameters
    ----------
    clf : sklearn classifier
        A sklearn or derived classifier or a pipeline.
    X_train : pd.DataFrame
        Containing the train attributes. Shape (n_samples_train, n_attributes).
    y_train : np.array
        Containing the class numbers in the train set. Shape (n_samples_train).
    X_test : pd.DataFrame
        Containing the test attributes. Shape (n_samples_test, n_attributes).
    y_test : np.array
        Containing the class numbers in the test set. Shape (n_samples_test).
    X_geom : gpd.GeoSeries
        Containing the geometries of the dataset.
    clf_name : str
        Name of the classifier to be displayed.
    combination : list
        List of the input features
    preprocessing : str
        Name of the preprocessing steps
    clf_type : sklearn classifier
        The classifier part of the pipeline
    mean_cv_accuracy : str
        The mean cross validation accuracy obtained, legacy
    comment : str
        Any kind of comment to add to the save file

    Returns
    -------
    clf : sklearn classifier
        the trained classifier
    y_pred : np.array
        The predicted classes of test set
    """
    y_train_pred, train_time, y_pred, test_time, clf = train_test(
            clf,
            X_train,
            y_train,
            X_test,
            y_test
            )
    
    return metrics_and_save_results(
            clf,
            y_train_pred,
            y_train,
            y_pred,
            y_test,
            X_geom,
            clf_name,
            combination,
            preprocessing,
            clf_type,
            mean_cv_accuracy,
            comment,
            train_time,
            test_time
            )
    
def metrics_and_save_results(
        clf,
        y_train_pred,
        y_train,
        y_pred,
        y_test,
        X_geom,
        clf_name,
        combination,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment,
        train_time,
        test_time
        ):
    """Compute the different metrics and save the model."""

    train_set_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    (train_set_precision,
     train_set_recall,
     train_set_f1,
     __) = sklearn.metrics.precision_recall_fscore_support(y_train,
                                                         y_train_pred,
                                                         average='macro',
                                                         zero_division=0)
    train_set_conf_mat = sklearn.metrics.confusion_matrix(y_train,
                                                          y_train_pred,
                                                          labels=np.arange(
                                                              len(US_utilises))
                                                          )
    print("Accuracy on the train set :", train_set_accuracy)

    

    
    test_set_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test,
                                                        y_pred,
                                                        labels=np.arange(
                                                            len(US_utilises))
                                                        )

    
    recalls, precisions, F1 = per_class_metrics(confusion_matrix)
    test_set_precision = np.nanmean(precisions)
    test_set_recall = np.nanmean(recalls)
    test_set_f1 = np.nanmean(F1)
    
    print("Accuracy on the test set :", test_set_accuracy)
    print("mF1 on the test set :", test_set_f1)
    print("Confusion matrix on the test set :\n", confusion_matrix)

    
    surf_classif_report = sklearn.metrics.classification_report(
        y_test,
        y_pred,
        sample_weight=X_geom.loc[X_test.index].area,
        zero_division=0,
        labels=np.arange(
            len(US_utilises)))
    
    #Plot the results
    df_matrix = pd.DataFrame(confusion_matrix,
                             index=US_utilises,
                             columns=US_utilises)
    recall_matrix_with_sep_diag(df_matrix)
    precision_matrix_with_sep_diag(df_matrix)
    F1_recall_precision_barh(F1, recalls, precisions, US_utilises, confusion_matrix)
    
    
    #Save the results
    new_results = [
        clf_name,
        str(combination),
        preprocessing,
        str(clf_type),
        mean_cv_accuracy,
        train_set_accuracy,
        train_set_recall,
        train_set_precision,
        train_set_f1,
        train_set_conf_mat,
        test_set_accuracy,
        test_set_recall,
        test_set_precision,
        test_set_f1,
        confusion_matrix
    ] + list(recalls) + list(precisions) + list(F1) + [
        surf_classif_report,
        train_time,
        test_time,
        str(datetime.datetime.now()),
        comment
        ]
    
    df = pd.read_excel(save_path, 0, index_col=0)
    if len(df.columns)==0:
        df = pd.DataFrame(
            columns=[
            'classifier name',
             'input features',
             'preprocessing',
             'best params',
             'mean cv accuracy',
             'train set accuracy',
             'train set recall',
             'train set precision',
             'train set F1',
             'train set confusion matrix',
             'test set accuracy',
             'test set recall',
             'test set precision',
             'test set F1',
             'test set confusion matrix'
            ] + [
                f"Rappel test {USi}"
                for USi in US_utilises
            ] + [
                f"Precision test {USi}"
                for USi in US_utilises
            ] + [
                f"F1 test {USi}"
                for USi in US_utilises
            ] + [
            'test_surf_classif_report',
            'train time', 'test time',
            'date', 'Comment'
                ]
            )
    L_results = df.to_numpy(dtype="object")
    if L_results.size != 0:
        L_results = np.vstack((
            L_results,
            new_results
            ))
    else:
        L_results = np.array(new_results).reshape(-1, 1).T
    
    pd.DataFrame(L_results,
                 columns=df.columns).to_excel(save_path,
                                              freeze_panes = (1, 5))
    print("\n")
    return clf, y_pred

#%%Defining objective, input and save path
if __name__ == "__main__":
    
    rng = np.random.RandomState(0)
    
    departement = 69# Choose between 32 for Gers and 69 for Rhône 
    objective = "all_LU" #Choose between "all_LU" to use all LU classes or "US235" to use only residential, secondary and tertiary production like in the article "Comparison of two data fusion approaches for land use classification", Cubaud et al, 2023
    sources = Sources(objective)
    Land_files_available = False
    
    if departement == 32:
        
        if objective == "US235":
    
            save_path = os.path.join(
                "Results_Gers",
                "US235_approach1",#1st approach with respect to the previous article
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "OCS_GE_Gers_LU235.gpkg",
                driver="GPKG").set_index("ID_1")
            
        if objective == "all_LU":
            
            save_path = os.path.join(
                "Results_Gers",
                "all_LU_approach1",
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "OCS_GE_Gers.gpkg",
                driver="GPKG").set_index("ID_1")
    
    elif departement == 69:
        
        if objective == "US235":
        
            save_path = os.path.join(
                "Results_Rhone",
                "US235_approach1",
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "OCS_GE_Rhone_LU235.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")
            
        if objective == "all_LU":
            
            save_path = os.path.join(
                "Results_Rhone",
                "all_LU_approach1",
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "OCS_GE_Rhone.gpkg",
                driver="GPKG").set_index("ID_1")
            
        ocsge = ocsge.rename(columns={
            "CODE_18" : "CODE_12",
            "CODE_18_mean_1m" : "CODE_12_mean_1m"
            })
        
    if objective == "US235":
        #If the objective is US235, we select only the polygons with LU class 2, 3 or 5
        subset_boolean_index = ocsge.loc[
            :, "CODE_US"
            ].isin(
                ['US2', 'US3', 'US5']
                )
    elif objective == "all_LU":
        #If the objective is all_LU,
        # we select all the polygons except those marked as LU235
        # This class LU235 is supposed to represent mixed uses
        # but actually was used in Gers dataset for unknown between LU2, LU3 or LU5 (or even sometimes LU1.1)
        subset_boolean_index = ~ocsge.loc[
            :, "CODE_US"
            ].isin(
                ['US235']
                )
                
    #Create the folder for save_path if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #Ordinally Encode some attributes
    ord_encoder = OrdinalEncoder(handle_unknown='error')
    old_names = [
        "CODE_CS",
         "CODE_CS_mean_1m",
         "TYP_IRIS", "TYP_IRIS_mean_1m",
         "OSO", "OSO_mean_1m",
         "CODE_12", "CODE_12_mean_1m",
         
     ]
    new_names = [
        "code_cs",
        "code_cs_mean_1m",
        "TYP_IRIS", "TYP_IRIS_mean_1m",
        "OSO", "OSO_mean_1m",
        "CODE_12", "CODE_12_mean_1m",
        "usage_dgfip", "usage_dgfip_mean_1m"
     ]
    
    if objective == "US235":
        old_names += ["usage", "usage_mean_1m"]
    if objective == "all_LU":
        old_names += ["land_files_main_LU", "land_files_main_LU_mean_1m"]
        old_names += ["NATURE_hydro", "NATURE_hydro_mean_1m"]
        new_names += ["NATURE_hydro", "NATURE_hydro_mean_1m"]
    
    ocsge[
        new_names
        ] = ord_encoder.fit_transform(
             ocsge.loc[
                 :,
                 old_names
                 ]
             )
    
    ocsge = ocsge.drop(
        columns=["CODE_CS",
                 "CODE_CS_mean_1m", "ID_1_mean_1m",
                 "CODE_US_mean_1m", "index_mean_1m",
                 "signature_mean_1m_mean_1m",
                 "usage", "usage_mean_1m"],
        errors='ignore')
    
    ocsge = ocsge.fillna(0)
    
    #%%Definition of the sources
    
    categorical_cols_list = sources.categorical_cols_list
    
    combinations = np.array(
        [
        sources.all_cols,
        sources.Geom_cols,
        sources.radiometric_cols,
        sources.CS_cols,
        sources.OSM_cols,
        sources.BD_TOPO_bati_cols,
        sources.BD_TOPO_autres_cols,
        sources.OSO,
        sources.CLC,
        sources.Foncier,
        sources.IRIS,
        sources.not_Geom_cols,
        sources.not_radiometric_cols,
        sources.not_CS_cols,
        sources.not_OSM_cols,
        sources.not_BD_TOPO_bati_cols,
        sources.not_BD_TOPO_autres_cols,
        sources.not_IRIS,
        sources.not_OSO,
        sources.not_CLC,
        sources.not_Foncier,
        sources.not_m1_cols,
        sources.not_CLC_neither_CS_cols
        ], dtype='object'
        )
    
    if objective == "all_LU":
        combinations = np.concatenate((
                    combinations,
                    np.array([
                        sources.RPG,
                        sources.not_RPG
                        ], dtype="object")
                    ))
    
    #%%Preprocessing blocks which are independent of the attributes
    
    rus_max_div_by_2 = RandomUnderSampler(random_state=0,
                             sampling_strategy=divide_max_by_2
                             )
    
    ros_min_1000 = RandomOverSampler(
        random_state=0,
        sampling_strategy=if_less_than_1000_set_1000
        )
    
    
    scaler = MinMaxScaler()
    scaler = scaler.set_output(transform="pandas")

     
    #%%Apply the classifier on each combination of sources
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )

    for combination in combinations:# Loop on the combiation of sources
        try:
            combination = list(set(combination) - set(sources.CS_cols))#We remove the CODE_CS column
            if not Land_files_available:
                combination = list(set(combination) - set(sources.Foncier))
            print(combination)
            columns_to_keep = list(combination) + ["CODE_US", "geometry"]
            ocsge_subset = ocsge.loc[
                subset_boolean_index,
                columns_to_keep
                ]
            y_text = ocsge_subset.loc[:, "CODE_US"]
            #Regroup some classes into the class Other
            y_text[y_text.isin(["US4.1.4", "US6.1", "US6.2", "US6.3", "US6.6"])] = "US_other"
            US_utilises, y = np.unique(
                y_text,
                return_inverse=True)
            X_geom = ocsge_subset.geometry
            X = ocsge_subset.loc[:, combination]
        
            categorical_features = X.columns.isin(categorical_cols_list)
            
            #Define the sub algorithm of SMOTE to use according to the presence of categorical features.
            if np.sum(categorical_features) == len(categorical_features):
                sm = SMOTEN(random_state=0)
            elif np.sum(categorical_features) == 0:
                sm = SMOTE(random_state=0)
            else:
                sm = SMOTENC(random_state=0,
                             categorical_features=categorical_features
                             )
            
            #Train 70%, eval 10%, test 20%
            #Eval set is not used, but kept for consistance with the DST approach of the first article
            train_size = 0.8
            eval_size = 0.1
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                train_size=train_size,
                                                                random_state=rng)
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                                  test_size=eval_size/train_size,
                                                                  random_state=rng)
            
            
            liste_clf = [#Here you can add more classifier to try and loop over them
                xgboost
                ]
            
            liste_clf_names = [#This list should be as long as the previous one
                "XGBoost"
                ]
            

            for i_clf, clf_type in enumerate(liste_clf):
                
                clf_name = liste_clf_names[i_clf]
                liste_pipelines = [#Here you can add several
                        Pipeline(steps=[
                            ("scaler", scaler),
                            ("ROS_to_1000", ros_min_1000),
                            ("rus_max_div_by_2", rus_max_div_by_2),
                            ("SMOTENC_sampler", sm),
                            ("clf", clf_type)
                        ])
                    ]
                sampling_names = [#This list should be as long as the previous one
                    'SMOTENC'
                    ]
                
                for j_sampling, clf in enumerate(liste_pipelines):
                    
                    preprocessing = f"minmax scaler, {sampling_names[j_sampling]}"
                    
                    clf, y_pred = train_test_and_save_result(
                        clf,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        X_geom,
                        clf_name,
                        combination,
                        np.array(clf.steps)[:-1, 0],
                        clf_type,
                        "NO CV",
                        comment=""
                        )
        except Exception as e:
            print(e)
            print(combination, "didn't work")

    #%% Saving the model
    sio.dump(clf, "clf_model_"+str(departement)+"_"+objective+".skops")