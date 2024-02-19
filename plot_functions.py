# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:44:17 2023

@author: MCubaud
"""
import os
import numpy as np
# import shapefile
import pandas as pd
import geopandas as gpd
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx

from color_f_US import nomenclature, color_from_US_or_CS
from definition_of_sources import Sources


COLOR_GERS="tab:blue"
COLOR_RHONE="tab:orange"
COLOR_F1="#4464AD"
COLOR_PA="#FAC748"
COLOR_UA="#ae62a2"

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

def plot_dataframe(df):
    fig, ax = plt.subplots(figsize=1.5*np.array(df.shape))
    plt.imshow(df, cmap='Wistia')
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
    plt.xlabel("prediction", fontsize=30)
    plt.yticks(ticks=range(len(df.index)), labels=df.index)
    plt.ylabel("ground truth", rotation=90, fontsize=30)
    plt.colorbar()
    M = ax.transData.get_matrix()
    #xscale = M[0,0]
    #yscale = M[1,1]
    #plt.scatter(np.where(df==0)[1], np.where(df==0)[0], marker='s', c='white', s=xscale**2, zorder=1000)
    for y in range(df.shape[0]):
       for x in range(df.shape[1]):
          plt.text(x , y, '%.4f' % df.iloc[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             zorder=10000
          )
    plt.show()

def plot_dataframe_with_separate_diag_color(df, title=""):
    fig, ax = plt.subplots(figsize=1.5*np.array(df.shape))
    #Diagonal elements
    ax.imshow(np.diag(np.diag(df)), cmap='Greens', vmin=0, vmax=1)
    #Off-Diagonal elements
    masked_conf_matrix = np.ma.masked_where(np.eye(df.shape[0], dtype=bool), df)
    ax.imshow(masked_conf_matrix, cmap="Reds", interpolation='nearest', vmin=0, vmax=1)
    # Set axis labels and title
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
    plt.xlabel("prediction", fontsize=30)
    plt.yticks(ticks=range(len(df.index)), labels=df.index)
    plt.ylabel("ground truth", rotation=90, fontsize=30)
    # Add color bars for both color maps
    cbar_diagonal = plt.colorbar(ax.get_images()[0], ax=ax, shrink=0.65, aspect=10)
    cbar_off_diagonal = plt.colorbar(ax.get_images()[1], ax=ax, shrink=0.65, aspect=10)
    # Set labels for color bars
    cbar_diagonal.set_label('Diagonal', rotation=90, labelpad=0)
    cbar_off_diagonal.set_label('Off-Diagonal', rotation=90, labelpad=0)
    cbar_diagonal.ax.yaxis.set_label_position('left')
    cbar_off_diagonal.ax.yaxis.set_label_position('left')
    M = ax.transData.get_matrix()
    #xscale = M[0,0]
    #yscale = M[1,1]
    #plt.scatter(np.where(df==0)[1], np.where(df==0)[0], marker='s', c='white', s=xscale**2, zorder=1000)
    for y in range(df.shape[0]):
       for x in range(df.shape[1]):
          plt.text(x , y, '%.4f' % df.iloc[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             zorder=10000
          )
    if title:
        plt.title(title)

def recall_matrix_with_sep_diag(df_matrix):
    plot_dataframe_with_separate_diag_color(
        (df_matrix.T/df_matrix.sum(axis=1)).T,
        "recall matrix"
        )
    plt.show()

def precision_matrix_with_sep_diag(df_matrix):
    plot_dataframe_with_separate_diag_color(
        df_matrix/df_matrix.sum(axis=0),
        "precision matrix"
        )
    plt.show()

def recall_matrix(df_matrix):
    fig, ax = plt.subplots(figsize=1.5*np.array(df_matrix.shape))
    df_normalised_per_row = (df_matrix.T/df_matrix.sum(axis=1)).T
    plt.imshow(df_normalised_per_row, cmap='Wistia')
    plt.xticks(ticks=range(len(df_matrix.columns)), labels=df_matrix.columns)
    plt.xlabel("prediction", fontsize=30)
    plt.yticks(ticks=range(len(df_matrix.index)), labels=df_matrix.index)
    plt.ylabel("ground truth", rotation=90, fontsize=30)
    plt.colorbar()
    M = ax.transData.get_matrix()
    xscale = M[0,0]
    yscale = M[1,1]
    #plt.scatter(np.where(df==0)[1], np.where(df==0)[0], marker='s', c='white', s=xscale**2, zorder=1000)
    for y in range(df_matrix.shape[0]):
       for x in range(df_matrix.shape[1]):
          plt.text(x , y, '%.4f' % df_normalised_per_row.iloc[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             zorder=10000
          )
    #plt.title("recall matrix")
    plt.show()

def precision_matrix(df_matrix):
    fig, ax = plt.subplots(figsize=1.5*np.array(df_matrix.shape))
    df_normalised_per_col = df_matrix/df_matrix.sum(axis=0)
    plt.imshow(df_normalised_per_col, cmap='Wistia')
    plt.xticks(ticks=range(len(df_matrix.columns)), labels=df_matrix.columns)
    plt.xlabel("prediction", fontsize=30)
    plt.yticks(ticks=range(len(df_matrix.index)), labels=df_matrix.index)
    plt.ylabel("ground truth", rotation=90, fontsize=30)
    plt.colorbar()
    M = ax.transData.get_matrix()
    xscale = M[0,0]
    yscale = M[1,1]
    #plt.scatter(np.where(df==0)[1], np.where(df==0)[0], marker='s', c='white', s=xscale**2, zorder=1000)
    #Write text in center of the cell
    for y in range(df_matrix.shape[0]):
       for x in range(df_matrix.shape[1]):
          plt.text(x , y, '%.4f' % df_normalised_per_col.iloc[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             zorder=10000
          )
    plt.show()

def F1_recall_precision_barh(F1, recall, precision, classes, conf_mat):
    n = len(classes)
    width = 0.25
    plt.barh(np.arange(n) + 2 * width, F1, width, label="F1", color=COLOR_F1)
    plt.barh(np.arange(n) + width, recall, width, label="PA", color=COLOR_PA)
    plt.barh(np.arange(n), precision, width, label="UA", color=COLOR_UA)
    plt.yticks(np.arange(n) + width, classes)
    plt.xticks(np.arange(0, 1.1, step=0.1))

    plt.legend(loc=(0, 1.02), ncols=3)
    for i in range(n):
        if np.sum(conf_mat[i])==0 and np.sum(conf_mat[:,i])==0:
            plt.annotate("Class absent and not predicted", [0.1, i+width], color="green", va="center")
        elif np.sum(conf_mat[:,i])==0 and recall[i]==0:
            plt.annotate("Class present but not predicted", [0.1, i+width], color="red", va="center")
        elif np.sum(conf_mat[i])==0 and precision[i]==0:
            plt.annotate("Class absent but predicted", [0.1, i+width], color="red", va="center")
    plt.gca().invert_yaxis()

def F1_recall_precision_barh_vs_baseline(
        F1, recall, precision,
        F1_baseline, recall_baseline, precision_baseline,
        classes, conf_mat):
    n = len(classes)
    baseline_style={
        "color":[0,0,0,0],
        "linewidth":1,
        "edgecolor":'k'
        }
    width = 0.25

    plt.barh(np.arange(n) + 2 * width, F1, width, label="F1", color=COLOR_F1)
    plt.barh(np.arange(n) + 2 * width, F1_baseline, width,
             label="F1 baseline", **baseline_style)
    plt.barh(np.arange(n) + width, recall, width, label="PA", color=COLOR_PA)
    plt.barh(np.arange(n) + width, recall_baseline, width,
             label="PA baseline", **baseline_style)
    plt.barh(np.arange(n), precision, width, label="UA",color=COLOR_UA)
    plt.barh(np.arange(n), precision_baseline, width,
             label="UA baseline", **baseline_style)

    plt.yticks(np.arange(n) + width, classes)
    plt.xticks(np.arange(0, 1.1, step=0.1))

    plt.legend(loc=(0, 1.02), ncols=3)
    for i in range(n):
        if np.sum(conf_mat[i])==0 and np.sum(conf_mat[:,i])==0:
            plt.annotate("Class absent and not predicted", [0.1, i], color="green", va="center")
        elif np.sum(conf_mat[:,i])==0 and recall[i]==0:
            plt.annotate("Class present but not predicted", [0.1, i], color="red", va="center")
        elif np.sum(conf_mat[i])==0 and precision[i]==0:
            plt.annotate("Class absent but predicted", [0.1, i], color="red", va="center")
    plt.gca().invert_yaxis()

def change_between_automatic_and_final_ocsge(df):

    df2 = (df.T/df.sum(axis=1)).T
    red = np.eye(len(df2))
    green = np.zeros(df2.shape)
    blue = np.ones(df2.shape)-np.eye(len(df2))
    colors = np.stack([red, green, blue])

    f = plt.figure(figsize=(20, 15))
    f.suptitle("Change in label between automatic and final OCSGE (Rhône 2020)")
    for i in range(len(df2)):
        ax = f.add_subplot(5, 5, i+1)
        df2.iloc[i].plot(kind='bar', title=f"From {df2.index[i]} (count={int(df.iloc[i].sum())}) to", ax=ax, color=colors.T[i])
        vmax = df2.iloc[i].max()
        for j in range(len(df2)):
            v = df2.iloc[i, j]
            if v>0.8*vmax:
                ax.text(j+0.5, v*0.95, f"{np.round(v,3)}", color=colors.T[i, j])
            elif v<=0.8*vmax:
                ax.text(j-0.3, v+0.02, f"{np.round(v,3)}", color=colors.T[i, j], rotation=90)

    # f = plt.figure(figsize=(20, 15))
    ax = f.add_subplot(5, 5, i+2)
    ax.axis('off')
    j = 0
    N = 17
    for code in nomenclature.index:
        if code in df2.index:
            j += 1
            if j>N//2:
                j=0
                ax = f.add_subplot(5, 5, i+3)
                ax.axis('off')
            color = nomenclature.loc[code,["R", "V", "B"]]/255
            ax.text(0, 1 - 2*j/N,
                    code,
                    color=round(1-color.sum()/3)*np.ones(3),
                    backgroundcolor=color,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    fontweight='bold')
            ax.text(0.18, 1 - 2*j/N, nomenclature.loc[code, "LIBELLE_EN"],
                    transform=ax.transAxes,
                    fontsize=7.5)


    f.tight_layout()
    plt.show()

def matrices_correlation_ocsge(ocsge, columns_to_keep):
    ocsge_subset = ocsge.loc[
        ocsge.loc[:, "CODE_US"].isin([
            'US2', 'US3', 'US5', 'US1.1']),
        columns_to_keep
        ]

    ocsge2 = ocsge.loc[
        (ocsge.loc[:, "CODE_US"]=='US2'),
        columns_to_keep
        ]

    ocsge3 = ocsge.loc[
        (ocsge.loc[:, "CODE_US"]=='US3'),
        columns_to_keep
        ]

    ocsge5 = ocsge.loc[
        (ocsge.loc[:, "CODE_US"]=='US5'),
        columns_to_keep
        ]

    # ocsge1 = ocsge.loc[
    #     (ocsge.loc[:, "CODE_US"]=='US1.1'),
    #     columns_to_keep
    #     ]

    #Affichage des matrices de corrélations
    for i, df in enumerate([ocsge2,
                            ocsge3,
                            ocsge5,
                            # ocsge1,
                            ocsge_subset]):
        plt.figure(figsize=(20,20))
        plt.imshow(df.select_dtypes(['number']).corr(),
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1)
        plt.xticks(ticks=(np.arange(len(df.select_dtypes(['number']).columns))),
                    labels=df.select_dtypes(['number']).columns,
                    rotation = 90)
        plt.yticks(ticks=(np.arange(len(df.select_dtypes(['number']).columns))),
                    labels=df.select_dtypes(['number']).columns)
        plt.colorbar();
        plt.title(
            "Matrice de corrélation des paramètres"
            f" pour US{[2, 3, 5, '2, US3 et US5'][i]}")

def correlation_matrices_per_source(ocsge):
    dos = Sources("all_LU")
    df = ocsge[dos.all_cols]
    df = df[df.columns[~ df.columns.str.contains("mean_1m")]].select_dtypes(['number'])
    corr = df.corr(method='spearman')
    plt.figure(figsize=(20,20))
    plt.imshow(corr,
                cmap="coolwarm",
                vmin=-1,
                vmax=1)
    plt.xticks(ticks=(np.arange(len(df.columns))),
                labels=df.columns,
                rotation = 90)
    plt.xticks(ticks=(0.5+ np.arange(len(corr.columns)-1)),
            minor=True
            )
    plt.yticks(ticks=(0.5+ np.arange(len(corr.columns)-1)),
        minor=True
        )
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.yticks(ticks=(np.arange(len(df.columns))),
                labels=df.columns)
    plt.colorbar()
    plt.grid(which='minor')
    #aggregate by source
    #remove the diagonal
    IQR = lambda x: x.quantile(0.75) - x.quantile(0.25)
    for aggregation in ["mean", 'std', "median", IQR, 'max', "min"]:
        corr_s = corr - np.diag([1]*len(corr))
        corr_s = corr_s.abs()
        corr_s["source"] = dos.dict_attributes_sources
        corr_s = corr_s.groupby("source").aggregate(aggregation)
        corr_s = corr_s.T
        corr_s["source"] = dos.dict_attributes_sources
        corr_s = corr_s.groupby("source").aggregate(aggregation)
        plt.figure(figsize=(20,20))
        plt.imshow(corr_s,
                    cmap="Reds",
                    vmin=0,
                    vmax=1)
        plt.xticks(ticks=(np.arange(len(corr_s.columns))),
                    labels=corr_s.columns,
                    rotation = 90)
        plt.xticks(ticks=(0.5+ np.arange(len(corr_s.columns)-1)),
                minor=True
                )
        plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.yticks(ticks=(np.arange(len(corr_s.columns))),
                    labels=corr_s.columns)
        plt.yticks(ticks=(0.5+ np.arange(len(corr_s.columns)-1)),
            minor=True
            )
        plt.colorbar(label=f"{aggregation} absolute correlation")
        plt.grid(which='minor')


def draw_metrics(path, rows, baseline_row=None, savepath=None):
    df = pd.read_excel(path)
    US_utilises = np.array(
        df.columns[
            df.columns.str.contains("F1 test")
            ].str.replace("F1 test ","").str.replace("US","LU").str.replace("_other","6")
        )
    sorter = US_utilises.argsort()
    US_sorted = US_utilises[sorter]
    if not isinstance(rows, (list, np.ndarray)):
        rows = [rows]
    if baseline_row is not None:
        baseline_conf_mat = str_matrix_to_numpy(
            df.loc[baseline_row, "test set confusion matrix"]
            )[sorter,:][:,sorter]
        (recalls_baseline,
         precisions_baseline,
         F1_baseline
         ) = per_class_metrics(baseline_conf_mat)
    for row in rows:
        if df.loc[row, "Comment"]:
            plt.figure()
            plt.title(df.loc[row, "Comment"])
            plt.show()
        str_mat = df.loc[row, "test set confusion matrix"]
        conf_mat = str_matrix_to_numpy(str_mat)[sorter,:][:,sorter]
        df_matrix = pd.DataFrame(
            conf_mat,
            index=US_sorted,
            columns=US_sorted
            )
        recalls, precisions, F1 = per_class_metrics(conf_mat)
        recall_matrix_with_sep_diag(df_matrix)
        precision_matrix_with_sep_diag(df_matrix)
        if baseline_row is None:
            F1_recall_precision_barh(
                F1, recalls, precisions, US_sorted, conf_mat
                )
        else:
            F1_recall_precision_barh_vs_baseline(
                F1, recalls, precisions,
                F1_baseline, recalls_baseline, precisions_baseline,
                US_sorted, conf_mat)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
    plt.show()


def importance_des_sources(df, rows, sources_names, savepath=None):
    """


    Example
    -------

    importance_des_sources(
        df,
        list(reversed([5, 6, 7, 11, 12, 9, 10, 27, 14, 13, 8, 26])),
        list(reversed(
            ["Geometry", "Radiometry",
             "OCSGE LC", "CLC", "OSO",
             "BD TOPO building", "BD TOPO other",
             "RPG", "INSEE", "Land Files",
             "OSM", "All sources"]
            ))
        )

    """
    columns = "test set F1"
    selected_metrics = df.loc[rows, columns]
    if "All sources" in sources_names:
        i = sources_names.index("All sources")
        colors = [COLOR_F1]*(len(rows))
        colors[i] = "green"
        if i!=len(rows)-1 and i!=0:
            colors[0]="red"
        plt.barh(
            sources_names,
            selected_metrics,
            color=colors
            )
        labels = plt.gca().get_yticklabels()
        labels[i].set_fontweight('bold')
        if i!=len(rows)-1:
            plt.plot([0, 1], [i+0.5, i+0.5], "k")
        if i!=0:
            plt.plot([0, 1], [i-0.5, i-0.5], "k")
    else:
        plt.barh(sources_names, selected_metrics)
    for i in range(len(sources_names)):
        score = selected_metrics.iloc[i]
        plt.text(score, i, f"{score :.2f}", ha='left', va='center')
    plt.ylabel("Sources")
    plt.xlabel("mF1")
    plt.xlim(0, 1)
    plt.ylim(-0.5, len(rows)-0.5)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
    plt.show()

def loco(df, rows, sources_names, baserow, savepath=None, xlims=None):
    """


    Example
    -------

    loco(
        df,
        list(reversed([15, 16, 17, 23, 22, 19, 20, 28, 21, 24, 18, 25])),
        list(reversed(
            ["Geometry", "Radiometry",
             "OCSGE LC", "CLC", "OSO",
             "BD TOPO building", "BD TOPO other",
             "RPG", "INSEE", "Land Files",
             "OSM", "Without neighbors means"]
            )),
        26
        )

    """
    columns = "test set F1"
    selected_metrics = df.loc[baserow, columns] - df.loc[rows, columns]
    #When we add a source to the baseline, we want to change the sign
    s = pd.Series(sources_names)
    selected_metrics.iloc[
        np.where(s.str.contains("\+"))[0]
        ] = -selected_metrics.iloc[np.where(s.str.contains("\+"))[0]]
    #Plotting
    plt.barh(sources_names, selected_metrics, color=COLOR_F1)
    plt.ylabel("Sources")
    plt.xlabel("mF1")
    #plt.xlim(-1, 1)
    plt.plot([0, 0], [-1, len(sources_names)+1], "k")
    plt.ylim(-0.5, len(rows)-0.5)
    if xlims is not None:
        plt.xlim(*xlims)
    for i in range(len(sources_names)):
        score = selected_metrics.iloc[i]
        if score<0:
            plt.text(score, i, f"{score :.3f}", ha='right', va='center')
        else:
            plt.text(score, i, f"{score :.3f}", ha='left', va='center')
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
    plt.show()

def per_class_metrics_in_function_of_class_size_from_row(
        class_size,
        df,
        row
        ):
    US_utilises = np.array(
        df.columns[
            df.columns.str.contains("F1 test")
            ].str.replace("F1 test ","").str.replace("US","LU").str.replace("_other","6")
        )
    rappels, precisions, f1s = per_class_metrics(str_matrix_to_numpy(df.loc[0, "test set confusion matrix"]))
    for x in [rappels, precisions, f1s]:
        x[np.isnan(x)] = 0

    class_size.index = class_size.index.str.replace("US", "LU")
    absent_classes = US_utilises[~np.isin(US_utilises, class_size.index)]
    class_size = class_size.T
    class_size[absent_classes] = 0
    class_size = class_size[US_utilises]
    class_size = np.array(class_size.T)
    per_class_metrics_in_function_of_class_size(
            class_size,
            f1s,
            rappels,
            precisions,
            US_utilises,
            #dep="Gers"
            )

def per_class_metrics_in_function_of_class_size(
        class_size,
        f1s,
        rappels,
        precisions,
        US_utilises,
        #dep="Gers"
        ):
    from scipy.stats import spearmanr
    r_f1, p_f1 = spearmanr(f1s, class_size)
    r_recall, p_recall = spearmanr(rappels, class_size)
    r_precision, p_precision = spearmanr(precisions, class_size)
    decalage = np.zeros((len(class_size)))
    #decalage[[3, 10, 8, 5, 15]] = [0.03, -0.01, -0.05, 0.01, -0.03]
    #decalage[[2, 7, 9, 11, 6, 4]] = [-0.02, 0.02, 0.02, -0.02, 0.02, -0.01]
    plt.plot(class_size, f1s, "o", label=f"F1 (Spearman correlation={r_f1:.2f}, p-value={p_f1:.2e})", color=COLOR_F1)
    plt.plot(class_size, precisions, "o", label=f"UA (Spearman correlation={r_precision:.2f}, p-value={p_precision:.2e})",color=COLOR_UA)
    plt.plot(class_size, rappels, "o", label=f"PA (Spearman correlation={r_recall:.2f}, p-value={p_recall:.2e})", color=COLOR_PA)
    for i in range(len(US_utilises)):
        plt.plot([class_size[i]]*2,[rappels[i], precisions[i]], "LightGrey", zorder=-1000)
        plt.annotate(US_utilises[i].replace("US", "LU"), (class_size[i], f1s[i]) + decalage[i])
    plt.xscale("log")
    plt.xlabel("Number of samples in dataset (log scale)")
    plt.ylabel("Metrics value")
    #plt.title(f"Per class metrics in function of class size in {dep}")
    plt.legend(loc=(0, 1.02))

def graph_confusion_matrix(conf_mat, US_utilises):
    precision_mat = conf_mat/conf_mat.sum(axis=0)
    precision_mat[np.isnan(precision_mat)] = 0
    precision_mat = precision_mat + precision_mat.T
    gnx = nx.from_numpy_array(precision_mat * (1-np.eye(len(precision_mat))))
    plt.figure(figsize=(35,35))
    ax = plt.subplot()
    pos = nx.layout.spring_layout(gnx)
    edge_labels = nx.get_edge_attributes(gnx, "weight")
    nx.draw_networkx_nodes(gnx, pos, ax=ax)
    nx.draw_networkx_edges(gnx, pos, width=100*np.array([edge_labels[key] for key in edge_labels]).astype(float), ax=ax)

    edge_labels = {key[:2] : f"{(edge_labels[key]):.2f}" for key in edge_labels}
    node_labels = {i:US for i, US in enumerate(US_utilises)}
    nx.draw_networkx_edge_labels(gnx, pos, edge_labels, label_pos=0.25, ax=ax)
    nx.draw_networkx_labels(gnx, pos, node_labels, ax=ax, font_color=[0.2, 0.2, 0.2])

def class_histogram():
    class_size_gers = pd.read_csv(
        os.path.join(
            "Documents",
            "Resultats",
            "all_LU_approche1",
            'class_size.csv'), index_col=0
        )
    class_size_rhone = pd.read_csv(
        os.path.join(
            "E:\\",
            "Resultats_69",
            "all_LU_approche1",
            'class_size.csv'), index_col=0
        )
    class_size_both_deps = pd.DataFrame(
        index=list(set(list(class_size_gers.index) + list(class_size_rhone.index))),
        columns=["Gers", "Rhône"]
        )

    class_size_both_deps["Gers"] = class_size_gers
    class_size_both_deps["Rhône"] = class_size_rhone

    class_size_both_deps = class_size_both_deps.sort_index().fillna(0)
    class_size_both_deps.index = class_size_both_deps.index.str.replace("US", "LU")

    class_size_both_deps =  class_size_both_deps[ class_size_both_deps.index!="LU235"]

    plt.figure(figsize=(10, 4))
    plt.bar(class_size_both_deps.index,
            class_size_both_deps["Gers"],
            width=-0.2,
            align="edge",
            label='Gers',
            color=COLOR_GERS)
    plt.bar(class_size_both_deps.index,
            class_size_both_deps["Rhône"],
            width=0.2,
            align="edge",
            label='Rhône',
            color=COLOR_RHONE)
    plt.xlabel("OCS GE LU class")
    plt.ylabel("Number of samples")
    plt.legend(loc='upper left')
    plt.ylim(0, 260000)
    plt.yticks(np.arange(250001, step=25000))
    plt.grid()
    for i in range(len(class_size_both_deps)):
        for j, y in enumerate(class_size_both_deps.iloc[i]):
            plt.text(i, y+3000, int(y),
                     rotation=90, va='bottom',
                     ha=['right', 'left'][j])
    savepath=os.path.join("Pictures","article2","distribution_of_classes_in_both_departments.eps")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()

def mixedTrainSet(df, train_dep, test_dep):
    pattern = r'\d+\.\d+'
    result_table = pd.DataFrame()
    for index, row in df.iterrows():
        comment = str(row["Comment"])
        if comment.startswith("train set : "):
            print(comment)
            matches = re.findall(pattern, comment)
            print(matches)
            train_departement_prop = float(matches[0])/100
            test_departement_prop = float(matches[1])/100
            mF1 = row["test set F1"]
            result_table.loc[train_departement_prop, test_departement_prop] = mF1

    result_table = result_table.sort_index().sort_index(axis=1)
    print(result_table.to_latex(float_format="%.2f", bold_rows=True))

    X, Y = result_table.columns.values, result_table.index.values
    X, Y = np.meshgrid(X, Y)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, result_table.values, color='k')
    sc = ax.scatter(X.flatten(), Y.flatten(), result_table.values.flatten(), c=result_table.values.flatten(), cmap='viridis', alpha=1, s=100)
    plt.colorbar(sc)
    ax.view_init(elev=30, azim=235)

    # Set labels
    ax.set_xlabel(f'proportion of {test_dep}')
    ax.set_ylabel(f'proportion of {train_dep}')
    ax.set_zlabel(f'mF1 evaluated on {test_dep}')
    # Show the plot
    plt.tight_layout()
    plt.show()



#%%
if __name__ == "__main__":
    class_size_rhone = pd.read_csv(
        os.path.join(
            "E:\\",
            "Resultats_69",
            "all_LU_approche1",
            'class_size.csv'), index_col=0
        )
    class_size_gers = pd.read_csv(
        os.path.join(
            "Documents",
            "Resultats",
            "all_LU_approche1",
            'class_size.csv'), index_col=0
        )
    class_histogram()

    ############################################################# RHONE

    save_path = os.path.join(
        "E:\\",
        "Resultats_69",
        "all_LU_approche1",
        "results_ml_fused_US4_1_4_US6_1_US6_2_US6_3_US6_6.xlsx"
        )

    draw_metrics(save_path, 33, baseline_row=None, savepath=os.path.join("Pictures","article2","All_cols_rhone_per_class_scores.eps"))

    df = pd.read_excel(save_path)

    #Importance des sources Rhône:
    importance_des_sources(df,
        list(reversed([15, 16, 27, 26, 18, 19, 43, 21, 20, 18, 33, 17])),
        list(reversed(
        ["Geometry", "Radiometry",
         "CLC", "OSO",
         "BD TOPO building", "BD TOPO other",
         "RPG", "INSEE", "Land Files",
         "OSM", "All sources", "OCSGE LC"]
        )),
        savepath=os.path.join("Pictures","article2","score_1_source_rhone.eps")
        )


    #loco Rhône
    #! Radiometry is not OK !
    rows = list(reversed([32, 22, 39, 38, 35, 36, 44, 37, 40, 34, 12]))
    sources_names = list(reversed(
    ["All - Geometry", "All - Radiometry",
     "All - CLC", "All - OSO",
     "All - BD TOPO building", "All - BD TOPO other",
     "All - RPG", "All - INSEE", "All - Land Files",
     "All - OSM", "All + OCSGE LC"]
    ))
    baserow=33
    savepath=os.path.join("Pictures","article2","loco_rhone.eps")
    loco(df, rows, sources_names, baserow, savepath=savepath, xlims=(-0.012, 0.051))

    ############################################################# GERS

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
    df = pd.read_excel(save_path)

    draw_metrics(save_path, 46, baseline_row=None, savepath=os.path.join("Pictures","article2","All_cols_gers_per_class_scores.eps"))

    #Importance des sources Gers:
    importance_des_sources(df,
        list(reversed([47, 48, 52, 51, 50, 12, 65, 53, 13, 49, 46, 9])),
        list(reversed(
        ["Geometry", "Radiometry",
         "CLC", "OSO",
         "BD TOPO building", "BD TOPO other",
         "RPG", "INSEE", "Land Files",
         "OSM", "All sources", "OCSGE LC"]
        )),
        savepath=os.path.join("Pictures","article2","score_1_source_gers.eps")
        )

    #loco gers
    rows = list(reversed([15, 54, 61, 60, 57, 58, 64, 59, 62, 56, 4]))
    sources_names = list(reversed(
    ["All - Geometry", "All - Radiometry",
     "All - CLC", "All - OSO",
     "All - BD TOPO building", "All - BD TOPO other",
     "All - RPG", "All - INSEE", "All - Land Files",
     "All - OSM", "All + OCSGE LC"]
    ))
    baserow=46
    savepath=os.path.join("Pictures","article2","loco_gers.eps")

    loco(df, rows, sources_names, baserow, savepath=savepath, xlims=(-0.13, 0.23))

    #without minus
    importance_des_sources(df,
        list(reversed([15, 54, 61, 60, 57, 58, 64, 59, 62, 56, 46, 4])),
        list(reversed(
        ["All - Geometry", "All - Radiometry",
         "All - CLC", "All - OSO",
         "All - BD TOPO building", "All - BD TOPO other",
         "All - RPG", "All - INSEE", "All - Land Files",
         "All - OSM", "All sources", "All + OCSGE LC"]
        ))
        )

    ############################################################# RHONE to GERS

    save_path = os.path.join(
        "E:\\",
        "Resultats_transferabilite",
        "all_LU",
    	"69to32",
        "results_ml.xlsx"
        )
    df = pd.read_excel(save_path)

    draw_metrics(save_path, 0, baseline_row=None, savepath=os.path.join("Pictures","article2","All_cols_rhone_to_gers_per_class_scores.eps"))


    #loco rhone to gers
    rows = list(reversed([7, 8, 14, 13, 10, 11, 18, 12, 15, 9]))
    sources_names = list(reversed(
    ["All - Geometry", "All - Radiometry",
     "All - CLC", "All - OSO",
     "All - BD TOPO building", "All - BD TOPO other",
     "All - RPG", "All - INSEE", "All - Land Files",
     "All - OSM"]
    ))
    baserow=0
    savepath=os.path.join("Pictures","article2","loco_rhone_to_gers.eps")

    loco(df, rows, sources_names, baserow, savepath=savepath, xlims=(-0.05, 0.125))

    mixedTrainSet(df, "Rhône", 'Gers')

    per_class_metrics_in_function_of_class_size_from_row(
            class_size_rhone,
            df,
            baserow
            )

    per_class_metrics_in_function_of_class_size_from_row(
            class_size_gers,
            df,
            baserow
            )

    ############################################################# GERS to RHONE

    save_path = os.path.join(
        "E:\\",
        "Resultats_transferabilite",
        "all_LU",
    	"32to69",
        "results_ml.xlsx"
        )

    draw_metrics(save_path, 33, baseline_row=None, savepath=os.path.join("Pictures","article2","All_cols_gers_to_rhone_per_class_scores.eps"))
    df = pd.read_excel(save_path)

    #loco gers to rhone
    rows = list(reversed([53, 39, 48, 47, 42, 43, 51, 44, 47, 41]))
    sources_names = list(reversed(
    ["All - Geometry", "All - Radiometry",
     "All - CLC", "All - OSO",
     "All - BD TOPO building", "All - BD TOPO other",
     "All - RPG", "All - INSEE", "All - Land Files",
     "All - OSM"]
    ))
    baserow=33
    savepath=os.path.join("Pictures","article2","loco_gers_to_rhone.eps")

    loco(df, rows, sources_names, baserow, savepath=savepath, xlims=(-0.07, 0.08))

    mixedTrainSet(df.iloc[50:], "Gers", "Rhône")


    per_class_metrics_in_function_of_class_size_from_row(
            class_size_gers,
            df,
            baserow
            )

    per_class_metrics_in_function_of_class_size_from_row(
            class_size_rhone,
            df,
            baserow
            )

