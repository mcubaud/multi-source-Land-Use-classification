# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:52:57 2023

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

from sklearnex import patch_sklearn
patch_sklearn()
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
        comment,
        save_path
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
            X_test,
            y_test,
            X_geom,
            clf_name,
            combinaison,
            preprocessing,
            clf_type,
            mean_cv_accuracy,
            comment,
            train_time,
            test_time,
            save_path
            )



def metrics_and_save_results(
        clf,
        y_train_pred,
        y_train,
        y_pred,
        X_test,
        y_test,
        X_geom,
        clf_name,
        combinaison,
        preprocessing,
        clf_type,
        mean_cv_accuracy,
        comment,
        train_time,
        test_time,
        save_path
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

def compute_confidence(clf, X_test, y_pred):
    y_proba = clf.predict_proba(X_test)
    entropy = -np.nansum(y_proba*np.log2(y_proba),axis=1)
    pb_sort = np.sort(y_proba, axis=1)
    dif_proba_1_2 = pb_sort[:, 2] - pb_sort[:, 1]

    N = len(US_utilises)
    f = plt.figure(figsize=(21, 21))
    ax = f.add_subplot(N, N, 1)
    ax.set_ylim(0,1)
    ax.set_xlim(0.3,1)
    # ax.set_yticks([])
    for i, usa in enumerate(US_utilises):
        for j, usb in enumerate(US_utilises):
            ax = f.add_subplot(N, N, j+N*i+1,
                                sharex=ax,
                                sharey=ax)
            bins = np.arange(11)/10
            cond = (y_test==i) & (y_pred==j)
            height, __ = np.histogram(
                y_proba[cond, j],
                bins,
                )
            if (cond.sum()>0):
                height = height/cond.sum()
            plt.bar(
                bins[:-1]+0.05,
                height,
                width=0.1
                )
            ax.set_title(f"{usa} -> prédit {usb}")
    f.suptitle("Histogrammes normalisés des taux de confiance du classifieur")
    plt.savefig(os.path.join(
    "Documents", "Resultats",
    "US235_approche1", "erreurs", "confiance.png"
    ))
    plt.show()

def surface_col_to_surface_fraction_col(X, cols):
    X[cols] = X[cols].to_numpy() / X["surface"].to_numpy().reshape(-1, 1)
    return X

def surface_fraction_col_to_surface_col(X, cols):
    X[cols] = X[cols].to_numpy() * X["surface"].to_numpy().reshape(-1, 1)
    return X

def allign_y(y_from, y_to, US_utilises_from, US_utilises_to):

    US_utilises_both = np.array(
        list(US_utilises_from) +
        list(
            set(US_utilises_to) - set(US_utilises_from)
        )
    )#this to ensure that the beggining has the same order than US_utilises_from
    #We allign the values to be the same

    y_to_alligned = np.empty_like(y_to)
    for i, index in enumerate(y_to):
        y_to_element = US_utilises_to[index]
        aligned_index = np.where(np.array(US_utilises_both) == y_to_element)[0]
        y_to_alligned[i] = aligned_index[0] if aligned_index.size > 0 else -1
        #A priori it will never contains -1

    return y_from, y_to_alligned, US_utilises_both

def get_X_Y(ocsge, subset_boolean_index):

    ocsge_subset = ocsge.loc[
        subset_boolean_index,
        columns_to_keep
        ]
    y_text = ocsge_subset.loc[:, "CODE_US"]
    y_text[y_text.isin(["US4.1.4", "US6.1", "US6.2", "US6.3", "US6.6"])] = "US_other"
    US_utilises, y = np.unique(
        y_text,
        return_inverse=True)
    X_geom = ocsge_subset.geometry

    X = ocsge_subset.loc[:, combinaison]

    return X, X_geom, y, y_text, US_utilises

#%%
if __name__ == "__main__":
    pass
#%%Defining objective, input and save path

    rng = np.random.RandomState(0)

    from_departement_to = "69to32"
    objective = "all_LU"
    sources = Sources(objective)

    save_path = os.path.join(
        "E:\\",
        "Resultats_transferabilite",
        objective,
        from_departement_to,
        "results_ml.xlsx"
        )

    if from_departement_to == "32to69":

        if objective == "US235":

            ocsge_from = gpd.read_file(
                "plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")

            ocsge_to = gpd.read_file(
                "E:\\plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")

        if objective == "all_LU":
            print("ocsge_from")
            ocsge_from = gpd.read_file(
                "plus_foncier_updated.gpkg",
                driver="GPKG").set_index("ID_1")
            print("ocsge_to")
            ocsge_to = gpd.read_file(
                "E:\\osm_lines.gpkg",
                driver="GPKG").set_index("ID_1")
            print("imported")

        ocsge_to = ocsge_to.rename(columns={
            "CODE_18" : "CODE_12",
            "CODE_18_mean_1m" : "CODE_12_mean_1m"
            })


    elif from_departement_to == "69to32":

        if objective == "US235":

            ocsge_to = gpd.read_file(
                "plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")

            ocsge_from = gpd.read_file(
                "E:\\plus_image.gpkg",
                layer='plus_image',
                driver="GPKG").set_index("ID_1")

        if objective == "all_LU":
            ocsge_from = gpd.read_file(
                "E:\\osm_lines.gpkg",
                driver="GPKG").set_index("ID_1")

            ocsge_to = gpd.read_file(
                "plus_foncier_updated.gpkg",
                driver="GPKG").set_index("ID_1")

        ocsge_from = ocsge_from.rename(columns={
            "CODE_18" : "CODE_12",
            "CODE_18_mean_1m" : "CODE_12_mean_1m"
            })


    if objective == "US235":

        def get_boolean_index(ocsge):
            return ocsge.loc[
                :, "CODE_US"
                ].isin(
                    ['US2', 'US3', 'US5']
                    )
    elif objective == "all_LU":
        def get_boolean_index(ocsge):
            return ~ ocsge.loc[
                :, "CODE_US"
                ].isin(
                    ['US235']
                    )

    subset_boolean_index_from = get_boolean_index(ocsge_from)
    subset_boolean_index_to = get_boolean_index(ocsge_to)
    # elif objective == "all_LU":
    #     subset_boolean_index = np.ones(len(ocsge), bool)


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

    def transform_ocsge(ocsge):
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
        return ocsge


    ocsge_from = transform_ocsge(ocsge_from)
    ocsge_to = transform_ocsge(ocsge_to)

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
        #sources.not_CLC_neither_CS_cols
        ], dtype='object'
        )

    if objective == "all_LU":
        combinaisons = np.hstack((
            combinaisons,
            [
                sources.RPG,
                sources.not_RPG
                ]
            ))

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


    scaler = MinMaxScaler() # StandardScaler() # DummyScaler() #
    scaler = scaler.set_output(transform="pandas")

    pca = PCA()

    #%%Un classifieur sur chaque source

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

    # svm = SVC(
    #     C=1,
    #     gamma=0.0001,
    #     kernel="rbf",
    #     random_state=0
    #     )

    for combinaison in combinaisons[11:12]:
    #for combinaison in [sources.all_cols]:
        try:
            combinaison = list(set(combinaison) - set(sources.CS_cols))
            print(combinaison)
            columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]


            X_train, X_geom_train, y_train, y_text_train, US_utilises_train = get_X_Y(
                ocsge_from,
                subset_boolean_index_from)
            # X_train, __, y_train, __ = train_test_split(X_train, y_train,
            #                                             train_size=0.8,
            #                                             random_state=rng,
            #                                             #stratify=y_train
            #                                             )
            X_test, X_geom_test, y_test, y_text_test, US_utilises_test = get_X_Y(
                ocsge_to,
                subset_boolean_index_to)
            # __, X_test, __, y_test = train_test_split(X_test, y_test,
            #                                         train_size=0.8,
            #                                         random_state=rng,
            #                                         stratify=y_test)

            y_train, y_test, US_utilises = allign_y(
                y_train, y_test,
                US_utilises_train, US_utilises_test)
            X_geom = X_geom_test

            categorical_features = X_train.columns.isin(categorical_cols_list)

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

            selector = SelectKBest(chi2, k=len(X_train.columns))

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
            # X_train, __, y_train, __ = train_test_split(X_from, y_from,
            #                                             train_size=0.8,
            #                                             random_state=rng)
            # __, X_test, __, y_test = train_test_split(X_to, y_to,
            #                                             train_size=0.8,
            #                                             random_state=rng)
            # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
            #                                                       test_size=0.1/0.8,
            #                                                       random_state=rng)


            liste_clf = [
                xgboost,
                # rf,
                #svm
                ]

            liste_clf_names = [
                "XGBoost",
                # "RF",
                #"SVM"
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
                        # Pipeline(steps=[
                        #     ("scaler", scaler),
                        #     ("SMOTENC_sampler", sm),
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


                for j_sampling, clf in enumerate(liste_pipelines):


                    train_test_and_save_result(
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
                        comment="",
                        save_path=save_path
                        )
        except Exception as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print(e)
            print(sources.find_source_name(combinaison), "didn't work")
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    #%%Un peu de chaque département dans le jeu de train

    save_path_2 = os.path.join(
        "E:\\",
        "Resultats_transferabilite",
        objective,
        {"32to69":"69to32", "69to32":"32to69"}[from_departement_to],
        "results_ml.xlsx"
        )

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

    # svm = SVC(
    #     C=1,
    #     gamma=0.0001,
    #     kernel="rbf",
    #     random_state=0
    #     )

    #for combinaison in combinaisons:
    for combinaison in [sources.all_cols]:

        columns_to_keep = list(combinaison) + ["CODE_US", "geometry"]

        iterations = 0
        for train_size_from in [0.05, 0.1]:#, 0.2, 0.5, 0.8]:
            for train_size_to in [0, 0.05, 0.1]:#, 0.2, 0.5, 0.8]:
                #iterations+=1
                #if iterations in [1]:
                for __ in range(5):
                    print("train size from :", train_size_from)
                    print("train size to :", train_size_to)
                    X_from, X_geom_from, y_from, y_text_from, US_utilises_from = get_X_Y(
                        ocsge_from,
                        subset_boolean_index_from)

                    if train_size_from!=0:
                        X_train_from, X_test_from, y_train_from, y_test_from = train_test_split(X_from, y_from,
                                                                    train_size=train_size_from,
                                                                    #random_state=rng,
                                                                    stratify=y_from
                                                                    )
                    else:
                        X_train_from = X_from[np.zeros_like(y_from, dtype=bool)]#shape (0, nb_dimension)
                        X_test_from = X_from
                        y_train_from = []
                        y_test_from = y_from

                    X_to, X_geom, y_to, y_text_test, US_utilises_to = get_X_Y(
                        ocsge_to,
                        subset_boolean_index_to)

                    y_train_from, y_to, US_utilises = allign_y(
                        y_train_from, y_to,
                        US_utilises_from, US_utilises_to)

                    if train_size_to!=0:
                        X_train_to, X_test, y_train_to, y_test = train_test_split(X_to, y_to,
                                                                train_size=train_size_to,
                                                                random_state=rng,
                                                                stratify=y_to)
                    else:

                        X_test = X_to
                        y_test = y_to
                        X_train_to = X_to[np.zeros_like(y_test, dtype=bool)]#shape (0, nb_dimension)
                        y_train_to = []

                    X_train = pd.concat([X_train_from, X_train_to])
                    y_train = np.concatenate([y_train_from, y_train_to])
                    print("total train size:", len(y_train))

                    categorical_features = X_train.columns.isin(categorical_cols_list)

                    if np.sum(categorical_features) == len(categorical_features):
                        # print("SMOTEN")
                        sm = SMOTEN(random_state=0)
                    elif np.sum(categorical_features) == 0:
                        # print("SMOTE")
                        sm = SMOTE(random_state=0)
                    else:
                        # print("SMOTENC")
                        sm = SMOTENC(random_state=0,
                                     categorical_features=categorical_features,
                                     categorical_encoder=None
                                     )

                    selector = SelectKBest(chi2, k=len(X_train.columns))

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
                    # X_train, __, y_train, __ = train_test_split(X_from, y_from,
                    #                                             train_size=0.8,
                    #                                             random_state=rng)
                    # __, X_test, __, y_test = train_test_split(X_to, y_to,
                    #                                             train_size=0.8,
                    #                                             random_state=rng)
                    # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                    #                                                       test_size=0.1/0.8,
                    #                                                       random_state=rng)


                    liste_clf = [
                        xgboost,
                        # rf,
                        #svm
                        ]

                    liste_clf_names = [
                        "XGBoost",
                        # "RF",
                        #"SVM"
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
                                # Pipeline(steps=[
                                #     ("scaler", scaler),
                                #     ("SMOTENC_sampler", sm),
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


                        for j_sampling, clf in enumerate(liste_pipelines):

                            y_train_pred, train_time, y_pred, test_time, clf = train_test(
                                    clf,
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test
                                    )

                            metrics_and_save_results(
                                    clf,
                                    y_train_pred,
                                    y_train,
                                    y_pred,
                                    X_test,
                                    y_test,
                                    X_geom,
                                    clf_name,
                                    combinaison,
                                    np.array(clf.steps)[:-1, 0],
                                    clf_type,
                                    "NO CV",
                                    f"train set : {train_size_from*100:.2f}% of train departement and {train_size_to*100:.2f}% of test departement",
                                    train_time,
                                    test_time,
                                    save_path
                                    )

                            t0 = time.time()
                            y_pred_from = clf.predict(X_test_from)
                            test_time = time.time() - t0

                            metrics_and_save_results(
                                    clf,
                                    y_train_pred,
                                    y_train,
                                    y_pred_from,
                                    X_test_from,
                                    y_test_from,
                                    X_geom_from,
                                    clf_name,
                                    combinaison,
                                    np.array(clf.steps)[:-1, 0],
                                    clf_type,
                                    "NO CV",
                                    f"train set : {train_size_to*100:.2f}% of train departement and {train_size_from*100:.2f}% of test departement",
                                    train_time,
                                    test_time,
                                    save_path_2
                                    )
                # except Exception as e:
                #     print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                #     print(e)
                #     print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

#%%
import matplotlib as mpl

deps = {"69":"Rhône", "32":"Gers"}

dep_from = deps[ from_departement_to[:2] ]
dep_to = deps[ from_departement_to[-2:] ]


df = pd.read_excel(save_path, 0, index_col=0)
liste_results = df.loc[:, "test set F1"]
liste_train_size_from = [0.05, 0.1, 0.2, 0.5, 0.8]
liste_train_size_to = [0, 0.005, 0.01, 0.05, 0.1, 0.2]
linestyles = ["-", "-.", "--", ":"]
marker = ['', 'o']
cmap = mpl.colormaps["rainbow"].resampled(len(liste_train_size_from))
n = len(liste_train_size_to)
for i in range(len(liste_train_size_from)):
    plt.plot(liste_train_size_to,
             liste_results.iloc[n*i:n*(i+1)],
             label=liste_train_size_from[i],
             color=cmap(i),
             linestyle=linestyles[i%len(linestyles)],
             marker=marker[i//len(linestyles)]
             )
#plt.ylim(0, 1)
plt.xlabel(f"Proportion of {dep_to} dataset in train set")
plt.ylabel("mF1")
plt.legend(title=f"Proportion of {dep_from} dataset in train set")
plt.title(f"Model evaluated on the rest of {dep_to} dataset")

plt.figure()
cmap = mpl.colormaps["rainbow"].resampled(len(liste_train_size_to))
n = len(liste_train_size_to)
for i in range(len(liste_train_size_to)):
    plt.plot(liste_train_size_from,
             liste_results[i::n],
             label=liste_train_size_to[i],
             color=cmap(i),
             linestyle=linestyles[i%len(linestyles)],
             marker=marker[i//len(linestyles)]
             )
#plt.ylim(0, 1)
plt.xlabel(f"Proportion of {dep_from} dataset in train set")
plt.ylabel("mF1")
plt.legend(title=f"Proportion of {dep_to} dataset in train set")
plt.title(f"Model evaluated on the rest of {dep_to} dataset")