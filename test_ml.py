#%% -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:25:22 2022

@author: MCubaud
"""

import os
import numpy as np
# import shapefile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import sklearn.metrics
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

#Imbalanced learn
from imblearn.over_sampling import SMOTENC, SMOTEN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from scipy import stats

import datetime
import time

from definition_of_sources import Sources
from plot_functions import recall_matrix_with_sep_diag, precision_matrix_with_sep_diag, F1_recall_precision_barh

def str_matrix_to_numpy(str_mat):
    str_mat = str_mat.replace("[", "").replace("]", "").split()
    n = int(len(str_mat)**0.5)
    return np.array(str_mat, dtype=float).reshape(n,n)

def per_class_metrics(conf_mat):
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

def complete_row(str_mat):
    conf_mat = str_matrix_to_numpy(str_mat)
    recalls, precisions, F1 = per_class_metrics(conf_mat)
    return list(np.concatenate((recalls, precisions, F1)))

def divide_max_by_factor(factor):
    def down_scaling(y):
        uniques, counts = np.unique(y, return_counts=True)
        counts[np.argmax(counts)] /= factor
        return {uniques[i]:counts[i] for i in range(len(uniques))}
    return down_scaling

def multiply_min_by_factor(factor):
    def up_scaling(y):
        uniques, counts = np.unique(y, return_counts=True)
        counts[np.argmin(counts)] *= factor
        return {uniques[i]:counts[i] for i in range(len(uniques))}
    return up_scaling

def if_less_than_value_set_value(value):
    def up_scaling(y):
        uniques, counts = np.unique(y, return_counts=True)
        counts[counts<value] = value
        return {uniques[i]:counts[i] for i in range(len(uniques))}
    return up_scaling

def sm_in_fonction_of_columns(columns):
    categorical_features = columns.isin(categorical_cols_list)
    
    if np.sum(categorical_features) == len(categorical_features):
        # print("SMOTEN")
        sm = SMOTEN(random_state=0)
    elif np.sum(categorical_features) == 0:
        # print("SMOTE")
        sm = SMOTE(random_state=0)
    else:
        # print("SMOTENC")
        sm = SMOTENC(random_state=0,
                     categorical_features=categorical_features
                     )
    
    return sm

def train_test(
        clf,
        X_train,
        y_train,
        X_test,
        y_test
        ):
    
    #TRAIN
    t0 = time.time()
    clf.fit(X_train, y_train)
    # clf.fit(X_train, y_train, clf__eval_set=[(X_eval, y_eval)])
    train_time = time.time() - t0
    y_train_pred = clf.predict(X_train)
    
    #TEST
    
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
        combinaison,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment
        ):
    
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
            combinaison,
            preprocessing,
            clf_type,
            mean_cv_accuracy,
            comment,
            train_time,
            test_time
            )

def train_test_and_save_result_models_built_up_not_built_up(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        X_geom,
        clf_name,
        combinaison,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment,
        is_built_up
        ):
    
    #MODEL FOR BUILT UP LAND COVER
    print("MODEL FOR BUILT UP LAND COVER")
    unique_y_train_built_up, inverse_y_train_built_up = np.unique(
        y_train[is_built_up[X_train.index]],
        return_inverse=True)
    y_train_built_up_reindexed = np.arange(
        len(unique_y_train_built_up)
        )[inverse_y_train_built_up]
    
    (y_train_pred_built_up, train_time_built_up,
    y_pred_built_up, test_time_built_up, clf) = train_test(
            clf,
            X_train[is_built_up],
            y_train_built_up_reindexed,
            X_test[is_built_up],
            y_test[is_built_up[X_test.index]]
            )
    y_train_pred_built_up = unique_y_train_built_up[y_train_pred_built_up]
    y_pred_built_up = unique_y_train_built_up[y_pred_built_up]
    
    #MODEL FOR NOT BUILT UP LAND COVER
    print("MODEL FOR NOT BUILT UP LAND COVER")
    unique_y_train_not_built_up, inverse_y_train_not_built_up = np.unique(
        y_train[~is_built_up[X_train.index]],
        return_inverse=True)
    y_train_not_built_up_reindexed = np.arange(
        len(unique_y_train_not_built_up)
        )[inverse_y_train_not_built_up]
    
    (y_train_pred_not_built_up, train_time_not_built_up,
     y_pred_not_built_up, test_time_not_built_up, clf) = train_test(
            clf,
            X_train[~is_built_up],
            y_train_not_built_up_reindexed,
            X_test[~is_built_up],
            y_test[~is_built_up[X_test.index]]
            )
         
    y_train_pred = reassemble_results(y_train,
                                      y_train_pred_built_up,
                                      y_train_pred_not_built_up,
                                      is_built_up[X_train.index])
    y_pred = reassemble_results(y_test,
                                y_pred_built_up,
                                y_pred_not_built_up,
                                is_built_up[X_test.index])
    train_time = train_time_built_up + train_time_not_built_up
    test_time = test_time_built_up + test_time_not_built_up
  
    return metrics_and_save_results(
            clf,
            y_train_pred,
            y_train,
            y_pred,
            y_test,
            X_geom,
            clf_name,
            combinaison,
            preprocessing,
            clf_type,
            mean_cv_accuracy,
            comment,
            train_time,
            test_time
            )

def train_test_and_save_result_models_for_groups_of_confused_classes(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        X_geom,
        clf_name,
        combinaison,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment,
        groups_of_confused_classes
        ):
    
    if groups_of_confused_classes == "auto":
        y_train_pred, train_time, y_pred, test_time, clf = train_test(
                clf,
                X_train,
                y_train,
                X_test,
                y_test
                )
        
        t0 = time.time()
        y_eval_pred = clf.predict(X_eval)
        eval_precision = sklearn.metrics.precision_score(y_eval, y_eval_pred, average=None)
        group = [label for label in range(len(eval_precision)) if eval_precision[label]<0.9]
        train_time += time.time()-t0
        print(group)
        bool_train = np.isin(y_train, group)
        bool_test = np.isin(y_pred, group)#We will repredict things that have been predicted in this group
        __, y_train_reindexed = np.unique(y_train[bool_train], return_inverse=True)
        __, y_test_reindexed = np.unique(y_test[bool_test], return_inverse=True)
        y_train_pred_group, train_time_group, y_pred_group, test_time_group, clf = train_test(
                clf,
                X_train[bool_train],
                y_train_reindexed,
                X_test[bool_test],
                y_test_reindexed
                )
        y_train_pred[bool_train] = np.array(group)[y_train_pred_group]
        y_pred[bool_test] = np.array(group)[y_pred_group]
        train_time += train_time_group
        test_time += test_time_group
        
    else:
    
        y_train_pred, train_time, y_pred, test_time, clf = train_test(
                clf,
                X_train,
                y_train,
                X_test,
                y_test
                )
        
        for group in groups_of_confused_classes:
            print(group)
            bool_train = np.isin(y_train, group)
            bool_test = np.isin(y_pred, group)#We will repredict things that have been predicted in this group
            __, y_train_reindexed = np.unique(y_train[bool_train], return_inverse=True)
            __, y_test_reindexed = np.unique(y_test[bool_test], return_inverse=True)
            y_train_pred_group, train_time_group, y_pred_group, test_time_group, clf = train_test(
                    clf,
                    X_train[bool_train],
                    y_train_reindexed,
                    X_test[bool_test],
                    y_test_reindexed
                    )
            y_train_pred[bool_train] = np.array(group)[y_train_pred_group]
            y_pred[bool_test] = np.array(group)[y_pred_group]
            train_time += train_time_group
            test_time += test_time_group
        
    return metrics_and_save_results(
            clf,
            y_train_pred,
            y_train,
            y_pred,
            y_test,
            X_geom,
            clf_name,
            combinaison,
            preprocessing,
            clf_type,
            mean_cv_accuracy,
            comment,
            train_time,
            test_time
            )

def reassemble_results(y, y_pred_1, y_pred_2, condition):
    y_pred_tot = np.ones_like(y)
    y_pred_tot[condition] = y_pred_1
    y_pred_tot[~condition] = y_pred_2
    return y_pred_tot
    
def metrics_and_save_results(
        clf,
        y_train_pred,
        y_train,
        y_pred,
        y_test,
        X_geom,
        clf_name,
        combinaison,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment,
        train_time,
        test_time
        ):

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
        str(combinaison),
        preprocessing,#downsample US3 and US5",#, downsample max by 2, SMOTENC, ",# pca",
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

def compute_confidence(clf, X_test, y_pred, y_test):
    y_proba = clf.predict_proba(X_test)
    entropy = -np.nansum(y_proba*np.log2(y_proba),axis=1)
    pb_sort = np.sort(y_proba, axis=1)[:, ::-1]
    proba_max = pb_sort[:, 0]
    dif_2_proba_max = pb_sort[:, 0] - pb_sort[:, 1]
        
    n_good = sum(y_pred==y_test)
    n_wrong = len(y_pred) - n_good
    weights = [
        np.ones(n_good) / len(y_pred),
        np.ones(n_wrong) / len(y_pred)
        ]
    for metric, name in [
            (proba_max, "Proba max"),
            (dif_2_proba_max, "Margin confidence"),
            (entropy, "Entropy")
            ]:
        r_p, p_p = stats.pearsonr(metric, y_pred==y_test)
        r_s, p_s = stats.spearmanr(metric, y_pred==y_test)
        #f = plt.figure(figsize=(21, 21))
        plt.figure()
        plt.hist(
            [
                metric[y_pred==y_test],
                metric[y_pred!=y_test]
                ],
            bins=20,
            weights=weights,
            stacked=True
        )
        #plt.hist(, label="y_pred==y_test")
        #plt.hist(proba_max[y_pred!=y_test], label="y_pred!=y_test")
        plt.legend([
            f"y_pred==y_test (min={np.min(metric[y_pred==y_test]):.2f}, max={np.max(metric[y_pred==y_test]):.2f})",
            f"y_pred!=y_test (min={np.min(metric[y_pred!=y_test]):.2f}, max={np.max(metric[y_pred!=y_test]):.2f})"]
            )
        plt.title(f"{name} (r_p={r_p:.2f}, p_p={p_p:.2e}, r_s={r_s:.2f}, p_s={p_s:.2e})")
    
    #N = len(US_utilises)
    # ax = f.add_subplot(N, N, 1)
    # ax.set_ylim(0,1)
    # ax.set_xlim(0.3,1)
    # # ax.set_yticks([])
    # for i, usa in enumerate(US_utilises):
    #     for j, usb in enumerate(US_utilises):
    #         ax = f.add_subplot(N, N, j+N*i+1,
    #                             sharex=ax,
    #                             sharey=ax)
    #         bins = np.arange(11)/10
    #         cond = (y_test==i) & (y_pred==j)
    #         height, __ = np.histogram(
    #             y_proba[cond, j],
    #             bins,
    #             )
    #         if (cond.sum()>0):
    #             height = height/cond.sum()
    #         plt.bar(
    #             bins[:-1]+0.05,
    #             height,
    #             width=0.1
    #             )
    #         ax.set_title(f"{usa} -> prédit {usb}")
    # f.suptitle("Histogrammes normalisés des taux de confiance du classifieur")
    # plt.savefig(os.path.join(
    # "Documents", "Resultats",
    # "US235_approche1", "erreurs", "confiance.png"
    # ))
    plt.show()
    
def surface_col_to_surface_fraction_col(X, cols):
    X[cols] = X[cols].to_numpy() / X["surface"].to_numpy().reshape(-1, 1)
    return X

def surface_fraction_col_to_surface_col(X, cols):
    X[cols] = X[cols].to_numpy() * X["surface"].to_numpy().reshape(-1, 1)
    return X

#%%Defining objective, input and save path
if __name__ == "__main__":
    
    rng = np.random.RandomState(0)
    
    departement = 69
    objective = "all_LU"
    sources = Sources(objective)
    
    if departement == 32:
        
        if objective == "US235":
    
            save_path = os.path.join(
                "Documents",
                "Resultats",
                "US235_approche1",
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")
            
        if objective == "all_LU":
            
            save_path = os.path.join(
                "Documents",
                "Resultats",
                "all_LU_approche1",
                #"results_ml_level_1.xlsx"
                "results_ml_fused_US4_1_4_US6_1_US6_2_US6_3_US6_6.xlsx"
                #"results_ml_all_except_US235.xlsx"#version with all the classes except LU235
                # "results_ml_all_classes.xlsx"#Old version with all the classes
                #"results_ml_fused_US4_1_4_US6_2_US6_3.xlsx"
                )
            
            ocsge = gpd.read_file(
                "plus_foncier_updated.gpkg",
                driver="GPKG").set_index("ID_1")
    
    elif departement == 69:
        
        if objective == "US235":
        
            save_path = os.path.join(
                "E:\\",
                "Resultats_69",
                "US235_approche1",
                "results_ml.xlsx"
                )
            
            ocsge = gpd.read_file(
                "E:\\plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")
            
        if objective == "all_LU":
            
            save_path = os.path.join(
                "E:\\",
                "Resultats_69",
                "all_LU_approche1",
                #"results_ml_level_1.xlsx"
                # "results_ml_all_classes.xlsx"#Old version with all the classes
                #"results_ml_fused_US4_1_4_US6_2_US6_3.xlsx"
                "results_ml_fused_US4_1_4_US6_1_US6_2_US6_3_US6_6.xlsx"
                )
            
            ocsge = gpd.read_file(
                "E:\\osm_lines.gpkg",
                driver="GPKG").set_index("ID_1")
            
        ocsge = ocsge.rename(columns={
            "CODE_18" : "CODE_12",
            "CODE_18_mean_1m" : "CODE_12_mean_1m"
            })
        
    if objective == "US235":
        subset_boolean_index = ocsge.loc[
            :, "CODE_US"
            ].isin(
                ['US2', 'US3', 'US5']
                )
    elif objective == "all_LU":
        #subset_boolean_index = np.ones(len(ocsge), bool)
        subset_boolean_index = ~ocsge.loc[
            :, "CODE_US"
            ].isin(
                ['US235']
                )
                
    CODE_CS = ocsge["CODE_CS"]
    
    # for column in ocsge.columns:
    #     if "Surf " in column:
    #         ocsge.loc[
    #             :,column
    #             ] = ocsge.loc[
    #                 :,column
    #                 ] / np.maximum(ocsge.loc[
    #                     :,"surface"
    #                     ],1)
    
    # __, ocsge.loc[:, "code_cs"] = np.unique(
    #     ocsge.loc[:, "CODE_CS"],
    #     return_inverse=True)
    
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
    
    ##Regrouper les machins de la BD TOPO
    # for aggr in ["Surf", "Nb"]:
    #     for end in ["", "_mean_1m"]:
    #         ocsge[f"{aggr}_US3_Bat_BD_topo{end}"] = ocsge.loc[
    #             :,
    #             (f"{aggr} Commercial et services{end}",
    #              f"{aggr} Religieux{end}",
    #              f"{aggr} Sportif{end}")].sum(axis=1)
    #         ocsge[f"{aggr}_inconnu_Bat_BD_topo{end}"] = ocsge.loc[
    #             :,
    #             (f"{aggr} Annexe{end}",
    #              f"{aggr} Indifférencié{end}"
    #              )].sum(axis=1)
    #         ocsge = ocsge.drop(
    #             columns=[f"{aggr} Commercial et services{end}",
    #                      f"{aggr} Religieux{end}",
    #                      f"{aggr} Sportif{end}",
    #                      f"{aggr} Annexe{end}",
    #                      f"{aggr} Indifférencié{end}"],
    #             errors='ignore')
            
    
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    
    # ocsge.loc[
    #     :,ocsge.loc[:,"CODE_CS"].unique()
    #     ] = pd.DataFrame(
    #         OH_encoder.fit_transform(
    #             ocsge.loc[:,"CODE_CS"].to_numpy().reshape(-1,1)),
    #         index=ocsge.index,
    #         columns=ocsge.loc[:,"CODE_CS"].unique()
    #         )
    
    # cs_mean_1m_new_cols = [val+"_mean_1m" for val in ocsge.loc[:,"CODE_CS_mean_1m"].unique()]
    # ocsge.loc[:,cs_mean_1m_new_cols] = pd.DataFrame(
    #     OH_encoder.fit_transform(
    #         ocsge.loc[:,"CODE_CS_mean_1m"].to_numpy().reshape(-1,1)),
    #     index=ocsge.index,
    #     columns=np.char.add(
    #         ocsge.loc[:,"CODE_CS_mean_1m"].unique().astype(str),
    #         "_mean_1m")
    #     )
    
    # #La TRICHE (°O°) mais si ça semble améliorer peut-être faire un procéssus en plusieurs étapes
    # ocsge.loc[:,ocsge.loc[:,"CODE_US_mean_1m"].unique()] = pd.DataFrame(
    #     OH_encoder.fit_transform(
    #         ocsge.loc[:,"CODE_US_mean_1m"].to_numpy().reshape(-1,1)),
    #     index=ocsge.index,
    #     columns=np.char.add(
    #         ocsge.loc[:,"CODE_US_mean_1m"].unique().astype(str),
    #         "_mean_1m")
    #     )
    
    
    geometry = ocsge.geometry
    
    
    ocsge = ocsge.drop(
        columns=["CODE_CS",
                 "CODE_CS_mean_1m", "ID_1_mean_1m",
                 "CODE_US_mean_1m", "index_mean_1m",
                 "signature_mean_1m_mean_1m",
                 "usage", "usage_mean_1m"],
        errors='ignore')
    
    ocsge = ocsge.fillna(0)
    # print(ocsge.max(axis=0, numeric_only=False))
    
    #%%Définition des sources
    
    categorical_cols_list = sources.categorical_cols_list
    
    combinaisons = np.array(
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
        combinaisons = np.concatenate((
            combinaisons,
            [
                sources.RPG,
                sources.not_RPG
                ]
            ), dtype="object")
    
    #%%Preprocessing blocks independent of the attributes
    
    
    rus_US5_div_by_2 = RandomUnderSampler(random_state=0,
                             sampling_strategy=divide_max_by_factor(2)
                             )
    
    rus_US5_taille_US3 = RandomUnderSampler(random_state=0,
                             sampling_strategy="majority"
                             )
    
    rus_US3_et_US5 = RandomUnderSampler(random_state=0
                             )
    
    ros_min_mult_by_10 = RandomOverSampler(random_state=0,
                             sampling_strategy=multiply_min_by_factor(10)
                             )
    
    ros_min_1000 = RandomOverSampler(
        random_state=0,
        sampling_strategy=if_less_than_value_set_value(1000)
        )
    
    
    scaler = MinMaxScaler() # DummyScaler() # StandardScaler() #
    scaler = scaler.set_output(transform="pandas")
    
    pca = PCA()
    
    #%%Version avec validation croisée
    
    for combinaison in [sources.all_cols]:#combinaisons:#
    # for combinaison in [list(X.columns[importances_mean>0])]:
        columns_to_keep = combinaison + ["CODE_US", "geometry"]
    
        ocsge_subset = ocsge.loc[
            subset_boolean_index,
            columns_to_keep
            ]
        
        y_text = ocsge_subset.loc[:, "CODE_US"]
        y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
        US_utilises, y = np.unique(
            y_text,
            return_inverse=True)
        X_geom = ocsge_subset.geometry
    
        X = ocsge_subset.loc[:, combinaison]
    
        class DummyScaler():
            def __init__(self):
                pass
            def fit(self, X):
                pass
            def transform(self, X):
                return np.array(X)
        
        #Train 70%, eval 10%, test 20%
        #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8,
                                                            random_state=rng,
                                                            stratify=y)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                              test_size=0.1/0.8,
                                                              random_state=rng,
                                                              stratify=y_train)
            
        #SMOTE
        sm = sm_in_fonction_of_columns(X.columns)
        
    
        selector = SelectKBest(chi2, k=len(X.columns))
    
        # X = SelectKBest(chi2, k=k).fit_transform(X, y)
        
        X_train = scaler.fit_transform(X_train, y_train)
        X_eval = scaler.transform(X_eval)
        X_test = scaler.transform(X_test)
        
        X_train, y_train = ros_min_1000.fit_resample(X_train, y_train)
        X_train, y_train = rus_US5_div_by_2.fit_resample(X_train, y_train)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    
        for (clf_part, param_grid, clf_name) in [
                # (RandomForestClassifier(random_state=0),
                #   [
                #       {'clf__n_estimators': [50, 100, 150, 500, 1000],
                #       'clf__max_features': ["sqrt", "log2", 0.2],
                #       'clf__max_depth': [None, 2, 5, 10, 20, 50]},
                #   ],
                # "RF"
                # ),
                # (XGBClassifier(random_state=0),
                #   [
                #       {'clf__n_estimators': [50, 100, 400, 700, 1000],
                #         'clf__learning_rate' : [0.5, 0.2, 0.1, 0.05, 0.02]}
                #       ],
                #   "XGBoost"
                #   ),
                # (RandomForestClassifier(random_state=0),
                #  [
                #      {'clf__n_estimators': [1000],
                #       'clf__max_features': [0.2],
                #       'clf__max_depth': [50]},
                #  ],
                # "RF"
                # ),
                # (XGBClassifier(random_state=0),
                #   [
                #       {'clf__n_estimators': [1000],
                #         'clf__learning_rate' : [0.2]}
                #       ],
                #   "XGBoost"
                #   ),
         
                (SVC(probability=True, random_state=0),
                  [
                      # {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear']},
                      {'clf__C': [0.1, 1, 10, 100, 1000], 'clf__gamma': [
                          0.1, 0.01, 0.001, 0.0001], 'clf__kernel': ['rbf']},
                      # {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['poly']},
                       ],
                  "SVM"
                  ),
                # (KNeighborsClassifier(),
                #   [
                #       {'clf__n_neighbors': [1, 5, 10, 15], 'clf__weights': [
                #           "uniform", "distance"]},
                #   ],
                # "KNN"
                # )
                ]:
    
            print(clf_name,"\n")
    
            # for dic in param_grid:
            #     dic["pca__n_components"]=[len(X.columns), 0.8, 0.9, 0.99, 'mle']
            
            
            pipe = Pipeline(steps=[
                # ("selector", selector),
                # ("scaler", scaler),
                # ("under_sampler", rus_US5_taille_US3),
                # ("under_sampler", rus),
                # ("SMOTENC_sampler", sm),
                # ("pca", pca),
                ("clf", clf_part)
                ])
    
    
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
            gridsearch = GridSearchCV(pipe, param_grid=param_grid,
                                      verbose=3, n_jobs=4, cv=cv,
                                      scoring={
                                           "accuracy" : sklearn.metrics.make_scorer(
                                               sklearn.metrics.accuracy_score),
                                           # "balanced_accuracy" : sklearn.metrics.make_scorer(
                                           #     sklearn.metrics.balanced_accuracy_score
                                           #     ),
                                          "f1_macro-mean" : sklearn.metrics.make_scorer(
                                              sklearn.metrics.f1_score,
                                              average="macro"
                                              ),
                                          },
                                      refit="f1_macro-mean")
            gridsearch.fit(X_train, y_train)
            best_params = gridsearch.best_params_
            print(best_params)
            mean_cv_accuracy = gridsearch.best_score_
            print(mean_cv_accuracy)
            cv_res = gridsearch.cv_results_
            clf = gridsearch.best_estimator_
            
            clf, y_pred = train_test_and_save_result(
                clf,
                X_train,
                y_train,
                X_test,
                y_test,
                X_geom,
                clf_name,
                combinaison,
                "Scaler, SMOTENC",#np.array(pipe.steps)[:-1, 0],
                clf_part,
                mean_cv_accuracy,
                comment=""
                )
    
    #%%Un classifieur sur chaque source
    print("Un classifieur sur chaque source")
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )
    
    # rf = RandomForestClassifier(
    #     max_depth=10,
    #     max_features='sqrt',
    #     n_estimators = 150,
    #     random_state=0
    #     )
    
    # rf = RandomForestClassifier(
    #     max_depth=50,
    #     max_features=0.2,
    #     n_estimators = 1000,
    #     random_state=0
    #     )
    
    # svm = SVC(
    #     C=1,
    #     gamma=0.0001,
    #     kernel="rbf",
    #     random_state=0
    #     )
    
    #for combinaison in combinaisons[[3, 7, 8]]:
    #for combinaison in combinaisons:
    for combinaison in [sources.not_radiometric_cols]:
    #for combinaison in [sources.all_cols]:
    # for combinaison in [
    #         sources.not_CLC_neither_CS_cols
    #         #list(set(sources.all_cols) - set(sources.CS_cols + sources.OSO + sources.CLC + sources.RPG))
    #         ]:
        try:
            combinaison = list(set(combinaison) - set(sources.CS_cols))
            print(combinaison)
            columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]
            print(1024)
            ocsge_subset = ocsge.loc[
                subset_boolean_index,
                columns_to_keep
                ]
            print(1029)
            y_text = ocsge_subset.loc[:, "CODE_US"]
            #y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
            y_text[y_text.isin(["US4.1.4", "US6.1", "US6.2", "US6.3", "US6.6"])] = "US_other"
            #y_text = pd.Series(y_text).str.slice(stop=3).to_numpy()
            print(1033)
            US_utilises, y = np.unique(
                y_text,
                return_inverse=True)
            print(1037)
            X_geom = ocsge_subset.geometry
            print(1039)
            X = ocsge_subset.loc[:, combinaison]
            
            print(X.shape)
        
            categorical_features = X.columns.isin(categorical_cols_list)
            
            if np.sum(categorical_features) == len(categorical_features):
                # print("SMOTEN")
                sm = SMOTEN(random_state=0)
            elif np.sum(categorical_features) == 0:
                # print("SMOTE")
                sm = SMOTE(random_state=0)
            else:
                # print("SMOTENC")
                sm = SMOTENC(random_state=0,
                             categorical_features=categorical_features
                             )
        
            selector = SelectKBest(chi2, k=len(X.columns))
        
            # X = SelectKBest(chi2, k=k).fit_transform(X, y)
            
            # clf = Pipeline(steps=[
            #     # ("selector", selector),
            #     ("scaler", scaler),
            #     # ("under_sampler", rus_US5_taille_US3),
            #     # ("under_sampler", rus_US3_et_US5),
            #     ("SMOTENC_sampler", sm),
            #     # ("pca", pca),
            #     # ("clf", rf)
            #     ("clf", xgboost)
            #     ])
        
            # clf_name = "XGBoost"
            # # clf_name = "RF"
            
            #Train 70%, eval 10%, test 20%
            #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                train_size=0.8,
                                                                random_state=rng)
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                                  test_size=0.1/0.8,
                                                                  random_state=rng)
            
            
            liste_clf = [
                xgboost,
                # rf,
                # svm
                ]
            
            liste_clf_names = [
                "XGBoost",
                # "RF",
                # "SVM"
                ]
            
        
            
            for i_clf, clf_type in enumerate(liste_clf):
                
                clf_name = liste_clf_names[i_clf]
                liste_pipelines = [
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("clf", clf_type)
                        # ]),
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("random_under_sampler", rus_US3_et_US5),
                        #     ("clf", clf_type)
                        # ]),
                        Pipeline(steps=[
                            ("scaler", scaler),
                            ("ROS_to_1000", ros_min_1000),
                            ("RUS_US5_div_by_2", rus_US5_div_by_2),
                            ("SMOTENC_sampler", sm),
                            ("clf", clf_type)
                        ])
                    
                    ]
                sampling_names = [
                    #'No sampling',
                    #'RUS US3 et US5',
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
                        combinaison,
                        np.array(clf.steps)[:-1, 0],
                        clf_type,
                        "NO CV",
                        comment=""
                        )
                    
                    #compute_confidence(clf, X_test, y_pred, y_test)
        except Exception as e:
            print(e)
            print(combinaison, "didn't work")
    #%%classifieurs selon la couverture bâti/non bâti
    
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )
    
    # rf = RandomForestClassifier(
    #     max_depth=10,
    #     max_features='sqrt',
    #     n_estimators = 150,
    #     random_state=0
    #     )
    
    rf = RandomForestClassifier(
        max_depth=50,
        max_features=0.2,
        n_estimators = 1000,
        random_state=0
        )
    
    svm = SVC(
        C=1,
        gamma=0.0001,
        kernel="rbf",
        random_state=0
        )
    
    # for combinaison in combinaisons:
    for combinaison in [sources.all_cols]:
        
        columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]
    
        ocsge_subset = ocsge.loc[
            subset_boolean_index,
            columns_to_keep
            ]
        
        is_built_up = (CODE_CS.loc[
            subset_boolean_index
            ] == "CS1.1.1.1")
        
        y_text = ocsge_subset.loc[:, "CODE_US"]
        #y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
        y_text[y_text.isin(["US4.1.4", "US6.1", "US6.2", "US6.3", "US6.6"])] = "US_other"
        US_utilises, y = np.unique(
            y_text,
            return_inverse=True)
        X_geom = ocsge_subset.geometry
    
        X = ocsge_subset.loc[:, combinaison]
        
        print(X.shape)
    
    
        categorical_features = X.columns.isin(categorical_cols_list)
        
        if np.sum(categorical_features) == len(categorical_features):
            # print("SMOTEN")
            sm = SMOTEN(random_state=0)
        elif np.sum(categorical_features) == 0:
            # print("SMOTE")
            sm = SMOTE(random_state=0)
        else:
            # print("SMOTENC")
            sm = SMOTENC(random_state=0,
                         categorical_features=categorical_features
                         )
    
        selector = SelectKBest(chi2, k=len(X.columns))
    
        # X = SelectKBest(chi2, k=k).fit_transform(X, y)
        
        # clf = Pipeline(steps=[
        #     # ("selector", selector),
        #     ("scaler", scaler),
        #     # ("under_sampler", rus_US5_taille_US3),
        #     # ("under_sampler", rus_US3_et_US5),
        #     ("SMOTENC_sampler", sm),
        #     # ("pca", pca),
        #     # ("clf", rf),
        #     ("clf", xgboost)
        #     ])
    
        # clf_name = "XGBoost"
        # # clf_name = "RF"
        
        #Train 70%, eval 10%, test 20%
        #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8,
                                                            random_state=rng)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                              test_size=0.1/0.8,
                                                              random_state=rng)
        
        
        liste_clf = [
            xgboost,
            # rf,
            # svm
            ]
        
        liste_clf_names = [
            "XGBoost",
            # "RF",
            # "SVM"
            ]
        
    
        
        for i_clf, clf_type in enumerate(liste_clf):
            
            clf_name = liste_clf_names[i_clf]
            liste_pipelines = [
                    # Pipeline(steps=[
                    #     ("scaler", scaler),
                    #     ("clf", clf_type)
                    # ]),
                    # Pipeline(steps=[
                    #     ("scaler", scaler),
                    #     ("random_under_sampler", rus_US3_et_US5),
                    #     ("clf", clf_type)
                    # ]),
                    Pipeline(steps=[
                        ("scaler", scaler),
                        ("ROS_to_1000", ros_min_1000),
                        ("RUS_US5_div_by_2", rus_US5_div_by_2),
                        ("SMOTENC_sampler", sm),
                        ("clf", clf_type)
                    ])
                
                ]
            sampling_names = [
                'No sampling',
                'RUS US3 et US5',
                # 'SMOTENC'
                ]
            
            for j_sampling, clf in enumerate(liste_pipelines):
                
                preprocessing = f"minmax scaler, {sampling_names[j_sampling]}"
                
                clf, y_pred = train_test_and_save_result_models_built_up_not_built_up(
                    clf,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    X_geom,
                    clf_name,
                    combinaison,
                    np.array(clf.steps)[:-1, 0],
                    clf_type,
                    "NO CV",
                    "2 classifiers Built-up/non Built-up",
                    is_built_up
                    )
                
                #compute_confidence(clf, X_test, y_pred, y_test)
    
        
    #%%classifieur secondaire sur certains groupes de classes
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )
    
    # rf = RandomForestClassifier(
    #     max_depth=10,
    #     max_features='sqrt',
    #     n_estimators = 150,
    #     random_state=0
    #     )
    
    rf = RandomForestClassifier(
        max_depth=50,
        max_features=0.2,
        n_estimators = 1000,
        random_state=0
        )
    
    svm = SVC(
        C=1,
        gamma=0.0001,
        kernel="rbf",
        random_state=0
        )
    
    # for combinaison in combinaisons:
    for combinaison in [sources.all_cols]:
        
        columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]
    
        ocsge_subset = ocsge.loc[
            subset_boolean_index,
            columns_to_keep
            ]
        y_text = ocsge_subset.loc[:, "CODE_US"]
        #y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
        y_text[y_text.isin(["US4.1.4", "US6.1", "US6.2", "US6.3", "US6.6"])] = "US_other"
        US_utilises, y = np.unique(
            y_text,
            return_inverse=True)
        X_geom = ocsge_subset.geometry
    
        X = ocsge_subset.loc[:, combinaison]
        
        print(X.shape)
    
    
        categorical_features = X.columns.isin(categorical_cols_list)
        
        if np.sum(categorical_features) == len(categorical_features):
            # print("SMOTEN")
            sm = SMOTEN(random_state=0)
        elif np.sum(categorical_features) == 0:
            # print("SMOTE")
            sm = SMOTE(random_state=0)
        else:
            # print("SMOTENC")
            sm = SMOTENC(random_state=0,
                         categorical_features=categorical_features
                         )
    
        selector = SelectKBest(chi2, k=len(X.columns))
    
        # X = SelectKBest(chi2, k=k).fit_transform(X, y)
        
        # clf = Pipeline(steps=[
        #     # ("selector", selector),
        #     ("scaler", scaler),
        #     # ("under_sampler", rus_US5_taille_US3),
        #     # ("under_sampler", rus_US3_et_US5),
        #     ("SMOTENC_sampler", sm),
        #     # ("pca", pca),
        #     # ("clf", rf)
        #     ("clf", xgboost)
        #     ])
    
        # clf_name = "XGBoost"
        # # clf_name = "RF"
        
        #Train 70%, eval 10%, test 20%
        #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8,
                                                            random_state=rng)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                              test_size=0.1/0.8,
                                                              random_state=rng)
        
        
        for list_groups in [
                #[["US1.1", "US1.2", "US2", "US3", "US4.1.2", "US4.1.3", "US4.3", "US5", "US_other"]],
                #[["US1.1", "US1.2", "US4.1.2", "US4.3", "US5", "US_other"], ["US2", "US3", "US5", "US4.1.3"]],
                "auto"
                ]:
            comment = "by group of classes "+str(list_groups)
            
            if list_groups != "auto":
                #transform to class number
                list_groups = [np.where(np.isin(US_utilises, group))[0] for group in list_groups]
        
        
            liste_clf = [
                xgboost,
                # rf,
                # svm
                ]
            
            liste_clf_names = [
                "XGBoost",
                # "RF",
                # "SVM"
                ]
            
        
            
            for i_clf, clf_type in enumerate(liste_clf):
                
                clf_name = liste_clf_names[i_clf]
                liste_pipelines = [
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("clf", clf_type)
                        # ]),
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("random_under_sampler", rus_US3_et_US5),
                        #     ("clf", clf_type)
                        # ]),
                        Pipeline(steps=[
                            ("scaler", scaler),
                            ("ROS_to_1000", ros_min_1000),
                            ("RUS_US5_div_by_2", rus_US5_div_by_2),
                            ("SMOTENC_sampler", sm),
                            ("clf", clf_type)
                        ])
                    
                    ]
                sampling_names = [
                    'No sampling',
                    'RUS US3 et US5',
                    # 'SMOTENC'
                    ]
                
                for j_sampling, clf in enumerate(liste_pipelines):
                    
                    preprocessing = f"minmax scaler, {sampling_names[j_sampling]}"
                    
                    clf, y_pred = train_test_and_save_result_models_for_groups_of_confused_classes(
                        clf,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        X_geom,
                        clf_name,
                        combinaison,
                        np.array(clf.steps)[:-1, 0],
                        clf_type,
                        "NO CV",
                        comment=comment,
                        groups_of_confused_classes=list_groups
                        )
                    
                    #compute_confidence(clf, X_test, y_pred, y_test)
    
    
    #%%Test fractions de surface
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )
    
    # rf = RandomForestClassifier(
    #     max_depth=10,
    #     max_features='sqrt',
    #     n_estimators = 150,
    #     random_state=0
    #     )
    
    # rf = RandomForestClassifier(
    #     max_depth=50,
    #     max_features=0.2,
    #     random_state=0
    #     )
    
    # for combinaison in combinaisons:
    for combinaison in [sources.all_cols]:
        
        columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]
    
        ocsge_subset = ocsge.loc[
            subset_boolean_index,
            columns_to_keep
            ]
        ocsge_subset = surface_fraction_col_to_surface_col(
            ocsge_subset,
            [
                'za_us1_1',
                 'za_us1_3',
                 'za_us1_4',
                 'za_us2',
                 'za_us3',
                 'za_us4_3',
                 'frac_surf_hydro',
                 'frac_surf_aerodromes',
                 'frac_surf_cimetieres'
                ]
            )
        y_text = ocsge_subset.loc[:, "CODE_US"]
        y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
        US_utilises, y = np.unique(
            y_text,
            return_inverse=True)
        X_geom = ocsge_subset.geometry
    
        X = ocsge_subset.loc[:, combinaison]
    
    
        categorical_features = X.columns.isin(categorical_cols_list)
        
        if np.sum(categorical_features) == len(categorical_features):
            # print("SMOTEN")
            sm = SMOTEN(random_state=0)
        elif np.sum(categorical_features) == 0:
            # print("SMOTE")
            sm = SMOTE(random_state=0)
        else:
            # print("SMOTENC")
            sm = SMOTENC(random_state=0,
                         categorical_features=categorical_features
                         )
    
        selector = SelectKBest(chi2, k=len(X.columns))
    
        # X = SelectKBest(chi2, k=k).fit_transform(X, y)
        
        # clf = Pipeline(steps=[
        #     # ("selector", selector),
        #     ("scaler", scaler),
        #     # ("under_sampler", rus_US5_taille_US3),
        #     # ("under_sampler", rus_US3_et_US5),
        #     ("SMOTENC_sampler", sm),
        #     # ("pca", pca),
        #     # ("clf", rf)
        #     ("clf", xgboost)
        #     ])
    
        # clf_name = "XGBoost"
        # # clf_name = "RF"
        
        #Train 70%, eval 10%, test 20%
        #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8,
                                                            random_state=rng)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                              test_size=0.1/0.8,
                                                              random_state=rng)
        
        
        liste_clf = [
            xgboost,
            # rf
            ]
        
        liste_clf_names = [
            "XGBoost",
            # "RF"
            ]
        
    
        
        for i_clf, clf_type in enumerate(liste_clf):
            
            clf_name = liste_clf_names[i_clf]
            liste_pipelines = [
                    # Pipeline(steps=[
                    #     ("scaler", scaler),
                    #     ("clf", clf_type)
                    # ]),
                    # Pipeline(steps=[
                    #     ("scaler", scaler),
                    #     ("under_sampler", rus_US3_et_US5),
                    #     ("clf", clf_type)
                    # ]),
                    Pipeline(steps=[
                        ("scaler", scaler),
                        ("ROS_to_1000", ros_min_1000),
                        ("RUS_US5_div_by_2", rus_US5_div_by_2),
                        ("SMOTENC_sampler", sm),
                        ("clf", clf_type)
                    ])
                
                ]
            sampling_names = [
                # 'No sampling',
                # 'RUS US3 et US5',
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
                    combinaison,
                    np.array(clf.steps)[:-1, 0],
                    clf_type,
                    "NO CV",
                    comment=""
                    )
    
    
    #%%
    
    
    def new_data(X_split, X_geom, clf):
        proba = clf.predict_proba(X_split)
        geom =  X_geom[X_geom.index.isin(X_split.index)].geometry.to_numpy()
        couche = gpd.GeoDataFrame(proba, geometry=geom)
        couche["ID"] = X_split.index
        couche_etendue = couche.copy()
        couche_etendue.geometry = couche.buffer(1)
        join = couche.overlay(couche_etendue,
                              how='intersection',
                              keep_geom_type=False)
        join.drop(np.where(join["ID_1"]==join["ID_2"])[0])
        aggfunc={}
        dtypes = join.dtypes
    
    
        weighted_mean = lambda x: np.average(x, weights=join.loc[x.index].area)
    
        def weighted_std(x):
            weights = join.loc[x.index].area
            average = np.average(x, weights=weights)
            variance = np.average((x-average)**2, weights=weights)
            return np.sqrt(variance)
    
        def biggest_neighbour(x):
            weights = join.loc[x.index].area
            i = np.argmax(weights)
            return x.iloc[i]
    
        for i, col in enumerate(join.columns):
            if col.endswith("2") :
                if np.issubdtype(dtypes[i], np.integer) or\
                    np.issubdtype(dtypes[i], np.floating):
                    aggfunc[col] = [
                        (col+"mean", weighted_mean),
                        (col+"std", weighted_std),
                        (col+"bg_n", biggest_neighbour)
                        ]
    
            # elif col!="geometry":
                # aggfunc[col]="first"
    
        aggr = join.groupby(
            by="ID_1").agg(
                aggfunc
            )
    
        # aggr[ couche.columns[1:]] = couche.set_index(index_name)
        couche = couche.set_index("ID")
        couche[ aggr.columns] = aggr
    
        couche = couche.drop(columns=[
            "geometry", "ID_2"
            ], errors="ignore").rename(
            columns=(lambda x: str(x)))
    
        # couche[X_split.columns]=X_split
        return couche
    
    def rule_prediction(X_2):
        pb_US_X_2 = X_2.loc[:,['0','1','2']]
        pb_sort = np.sort(pb_US_X_2)
        pb_argsort = np.argsort(pb_US_X_2, axis=1)
        diff = pb_sort[:,2]-pb_sort[:,1]
        y_rpred = pd.DataFrame(index=X_2.index)
    
        seuil = 0
    
        y_rpred.loc[
            X_2.index[np.where(diff>=seuil)[0]],
            "US_pred"] = np.array(['0','1','2'])[
                np.where(pb_argsort==2)[1]
                ][np.where(diff>=seuil)[0]]
    
        pb_bgn_X_2 = X_2.loc[:,[
            "('0_2', '0_2mean')",
            "('1_2', '1_2mean')",
            "('2_2', '2_2mean')"]].to_numpy()
        pb_bgn_X_2 += pb_US_X_2
        pb_bgn_argsort = np.argsort(pb_bgn_X_2, axis=1)
    
        y_rpred.loc[
            X_2.index[np.where(diff<seuil)[0]],
            "US_pred"] = np.array(['0','1','2'])[
                np.where(pb_bgn_argsort==2)[1]
                ][np.where(diff<seuil)[0]]
    
        return y_rpred.to_numpy()
    
    X_2 = new_data(X, X_geom, clf)
    X_2_train = new_data(X_train, X_geom, clf)
    X_2_test = new_data(X_test, X_geom, clf)
    
    clf_2 = XGBClassifier()
    param_grid = [
        {'clf__n_estimators': [200, 400, 700, 1000],
              'clf__learning_rate' : [0.5, 0.2, 0.1, 0.05, 0.02]}
        ]
    # clf_2 = SVC(probability=True)
    # param_grid = [
    #     {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear']},
    #     {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [
    #         0.001, 0.0001], 'clf__kernel': ['rbf']},
    #     {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['poly']},
    # ]
    
    pipe_2 = Pipeline(steps=[
        ("scaler", scaler),
        ("clf", clf_2)
        ])
    
    gridsearch = GridSearchCV(pipe_2, param_grid=param_grid,
                              verbose=3, n_jobs=2)
    # gridsearch.fit(X_2_train, y_train)
    gridsearch.fit(X_2, y)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)
    cv = gridsearch.cv_results_
    clf_2 = gridsearch.best_estimator_
    
    # clf_2.fit(X_2_train, y_train)
    # print("Accuracy on the train set :", clf_2.score(X_2_train, y_train))
    
    y_2_pred = clf_2.predict(X_2_test)
    print("Accuracy on the test set :",
          sklearn.metrics.accuracy_score(y_test, y_2_pred))
    
    print("Confusion matrix on the test set :\n",
          sklearn.metrics.confusion_matrix(y_test, y_2_pred))
    
    y_2_rpred = rule_prediction(X_2).astype(int)
    print("Accuracy with rule prediction :",
          sklearn.metrics.accuracy_score(y, y_2_rpred))
    
    print("Confusion matrix with rule prediction :\n",
          sklearn.metrics.confusion_matrix(y, y_2_rpred))
    
    #%%
    
    ocsge["US_predicted"] = np.array(US_utilises)[
        clf.predict(
            ocsge.loc[:, X.columns]
            )
        ]
    
    ocsge[
        ["proba US2", "proba US3", "proba US5"]
        ] = clf.predict_proba(
            ocsge.loc[:, X.columns]
        )
            
    US23ou5 = ocsge["CODE_US"].isin(US_utilises)
    
    total_accuracy = (ocsge.loc[
        US23ou5, "CODE_US"
        ] == ocsge.loc[
            US23ou5, "US_predicted"
            ]).sum()
    
    total_accuracy /= US23ou5.sum()
    
    print("Sur l'ensemble des polygones US2, 3 ou 5, \
          y compris ceux ayant servi à l'entrainement, \
          sans ré-équilibrage, l'accuracy est de :",
          total_accuracy)
    
    ocsge = gpd.GeoDataFrame(
        ocsge,
        geometry=geometry)
    
    ocsge.to_file(
        "prediction_ML.gpkg",
        layer='prediction_ML',
        driver="GPKG")
    
    #%%
    
    for k, US in enumerate(US_utilises):
        results = [stats.pearsonr(X.iloc[:,i], y==k) for i in range(len(X.columns))]
        cor = np.array([res.statistic for res in results])
        p_values =  np.array([res.pvalue for res in results])
        sorted_idx = cor.argsort()
        plt.figure(figsize=(25,25))
        plt.barh(
            np.array(X.columns)[sorted_idx],
            cor[sorted_idx]
            )
        plt.grid()
        plt.title(
            "Corrélation entre les paramètres et CODE_US=="+US)
        plt.savefig(os.path.join(
            "Documents", "Resultats",
            "US235_approche1", "Statistiques",
            "Corr_"+US+".png")
            )
        plt.figure(figsize=(25,25))
        plt.barh(
            np.array(X.columns)[sorted_idx],
            p_values[sorted_idx]
            )
        plt.grid()
        # plt.xscale("log")
        plt.title(
            "P-value de la corrélation entre les paramètres et CODE_US=="+US)
        plt.savefig(os.path.join(
            "Documents", "Resultats",
            "US235_approche1", "Statistiques",
            "p_value_"+US+".png")
            )
        
    #%%
    for col in X.columns:
        f = plt.figure(figsize=(21, 21))
        for i, usa in enumerate(US_utilises):
            for j, usb in enumerate(US_utilises):
                ax = f.add_subplot(3, 3, i+3*j+1)
                plt.hist(
                    ocsge.loc[
                        (ocsge["CODE_US"]==usa) &\
                        (ocsge["US_predicted"]==usb),
                        col]
                        )
                ax.set_title(f"{usa} -> prédit {usb}")
        f.suptitle(col)
        plt.savefig(os.path.join(
        "Documents", "Resultats",
        "US235_approche1", "erreurs", col+".png"
        ))
        plt.show()
    
    #%%
    
    US235_random_250 = gpd.read_file(
        os.path.join(
            "Documents",
            "Donnees",
            "OCS_GE",
            "IA_2019",
            "US235_250_random.gpkg")
        )
    
    X_US235_random_250 = ocsge.loc[US235_random_250.ID, X.columns]
    y_true_US235_random_250 = US235_random_250["US_unmixed"].fillna("unknown")
    y_pred_US235_random_250 = US_utilises[clf.predict(X_US235_random_250)]
    proba_US235_random_250 = clf.predict_proba(X_US235_random_250)
    
    bool_235 = y_true_US235_random_250.isin(US_utilises)
    
    print("Accuracy on US235 among those "
          f"manualy labeled {US_utilises} :",
          sklearn.metrics.accuracy_score(
              y_true_US235_random_250[bool_235],
              y_pred_US235_random_250[bool_235])
          )
    
    print("Confusion matrix on US235 among those "
          f"manualy labeled {US_utilises} :\n",
          sklearn.metrics.confusion_matrix(
              y_true_US235_random_250[bool_235],
              y_pred_US235_random_250[bool_235])
          )
    
    #%%
    to_change_sources = [
        #BD TOPO buildings
        {
            "features" : [
                'Surf Agricole', 'Surf Annexe',
                'Surf Commercial et services', 'Surf Sportif',
                'Surf Industriel', 'Surf Religieux', 'Surf Résidentiel',
                'Surf Indifférencié'
                ],
            "kind" : "surface"
            },
                
     #osm polygons
         {
             "features" : [
                'osm_LU1_1_surface',
                'osm_LU1_2_surface',
                'osm_LU2_surface',
                'osm_LU3_surface',
                'osm_LU4_1_1_surface',
                'osm_LU4_1_2_surface',
                'osm_LU4_1_3_surface',
                'osm_LU6_1_surface',
                'osm_LU4_3_surface',
                'osm_LU5_surface',
                'osm_LU6_2_surface'],
             "kind" : "surface"
            },
    #osm landuse polygons
        {
            "features" : [
                'osm_LU1_1_landuse_surface',
                'osm_LU2_landuse_surface',
                'osm_LU3_landuse_surface',
                'osm_LU5_landuse_surface'],
             "kind" : "surface"
            },
    #RPG
        {
             "features" : [
                'surf_RPG'],
             "kind" : "surface"
            },
    #BD TOPO Other
        {
             "features" : [
                'za_us1_1',
                 'za_us1_3',
                 'za_us1_4',
                 'za_us2',
                 'za_us3',
                 'za_us4_3',
                 'frac_surf_hydro',
                 'frac_surf_aerodromes',
                 'frac_surf_cimetieres'],
              "kind" : "surface_fraction"
        }
    ]
    
    #%%Influence de la taille du jeu d'entraînement
    
    xgboost = XGBClassifier(
        random_state=0,
        learning_rate=0.1, 
        n_estimators=1000
        )
    
    # rf = RandomForestClassifier(
    #     max_depth=10,
    #     max_features='sqrt',
    #     n_estimators = 150,
    #     random_state=0
    #     )
    
    # rf = RandomForestClassifier(
    #     max_depth=50,
    #     max_features=0.2,
    #     n_estimators = 1000,
    #     random_state=0
    #     )
    
    # svm = SVC(
    #     C=1,
    #     gamma=0.0001,
    #     kernel="rbf",
    #     random_state=0
    #     )
    
    # for combinaison in combinaisons:
    for combinaison in [sources.all_cols]:
        
        columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]
    
        ocsge_subset = ocsge.loc[
            subset_boolean_index,
            columns_to_keep
            ]
        y_text = ocsge_subset.loc[:, "CODE_US"]
        # y_text[y_text.isin(["US4.1.4", "US6.2", "US6.3"])] = "US_other"
        US_utilises, y = np.unique(
            y_text,
            return_inverse=True)
        X_geom = ocsge_subset.geometry
    
        X = ocsge_subset.loc[:, combinaison]
        
        print(X.shape)
    
    
        categorical_features = X.columns.isin(categorical_cols_list)
        
        if np.sum(categorical_features) == len(categorical_features):
            # print("SMOTEN")
            sm = SMOTEN(random_state=0)
        elif np.sum(categorical_features) == 0:
            # print("SMOTE")
            sm = SMOTE(random_state=0)
        else:
            # print("SMOTENC")
            sm = SMOTENC(random_state=0,
                         categorical_features=categorical_features
                         )
    
        selector = SelectKBest(chi2, k=len(X.columns))
    
        # X = SelectKBest(chi2, k=k).fit_transform(X, y)
        
        # clf = Pipeline(steps=[
        #     # ("selector", selector),
        #     ("scaler", scaler),
        #     # ("under_sampler", rus_US5_taille_US3),
        #     # ("under_sampler", rus_US3_et_US5),
        #     ("SMOTENC_sampler", sm),
        #     # ("pca", pca),
        #     # ("clf", rf)
        #     ("clf", xgboost)
        #     ])
    
        # clf_name = "XGBoost"
        # # clf_name = "RF"
        
        #Train 70%, eval 10%, test 20%
        #Eval ne sert à rien mais c'est pour que ce soit pareil que dans DST
        for n in range(1, 2):
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                train_size=n/1500,
                                                                random_state=rng)
            # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
            #                                                       test_size=0.1/0.8,
            #                                                       random_state=rng)
            
            
            liste_clf = [
                xgboost,
                # rf,
                # svm
                ]
            
            liste_clf_names = [
                "XGBoost",
                # "RF",
                # "SVM"
                ]
            
        
            
            for i_clf, clf_type in enumerate(liste_clf):
                
                clf_name = liste_clf_names[i_clf]
                liste_pipelines = [
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("clf", clf_type)
                        # ]),
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("random_under_sampler", rus_US3_et_US5),
                        #     ("clf", clf_type)
                        # ]),
                        Pipeline(steps=[
                            ("scaler", scaler),
                            # ("ROS_to_1000", ros_min_1000),
                            # ("RUS_US5_div_by_2", rus_US5_div_by_2),
                            ("SMOTENC_sampler", sm),
                            ("clf", clf_type)
                        ])
                    
                    ]
                sampling_names = [
                    'No sampling',
                    'RUS US3 et US5',
                    # 'SMOTENC'
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
                        combinaison,
                        np.array(clf.steps)[:-1, 0],
                        clf_type,
                        "NO CV",
                        comment=f"train size {n}/1500"
                        )
    