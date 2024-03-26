#%% -*- coding: utf-8 -*-
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
    """The function `str_matrix_to_numpy` converts a string representation of a matrix into a NumPy array
    of floating-point numbers.
    
    Parameters
    ----------
    str_mat
        The `str_mat` parameter is a string representation of a matrix in the format of a 2D list, similar to what is displayed by the print function.
        eg for a 2x2 matrix: 
            "[[1 0]
            [0 1]]"
    Returns
    -------
        This function takes a string representation of a matrix, converts it to a numpy array of floats,
    and reshapes it into a square matrix. The function returns the numpy array representing the matrix.
    
    """
    str_mat = str_mat.replace("[", "").replace("]", "").split()
    n = int(len(str_mat)**0.5)
    return np.array(str_mat, dtype=float).reshape(n,n)

def per_class_metrics(conf_mat):
    """The function calculates precision, recall, and F1 score for each class based on a confusion matrix.
    
    Parameters
    ----------
    conf_mat
        The function `per_class_metrics` calculates precision, recall, and F1 score for each class based on
    the confusion matrix provided as input. The confusion matrix `conf_mat` should represent the true
    positive, false positive, true negative, and false negative values for each class.
    
    Returns
    -------
        The function `per_class_metrics` returns three arrays: `recalls`, `precisions`, and `F1`. These
    arrays contain the recall, precision, and F1 score metrics calculated for each class based on the
    input confusion matrix `conf_mat`.
    
    """
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
    """The `plot_dataframe` function creates a heatmap plot of a DataFrame with annotations showing the
    values of each cell.
    
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe representation of the confusion matrix, with class labels as index and columns
    
    """
    __, ax = plt.subplots(figsize=1.5*np.array(df.shape))
    plt.imshow(df, cmap='Wistia')
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
    plt.xlabel("prediction", fontsize=30)
    plt.yticks(ticks=range(len(df.index)), labels=df.index)
    plt.ylabel("ground truth", rotation=90, fontsize=30)
    plt.colorbar()
    M = ax.transData.get_matrix()
    for y in range(df.shape[0]):
       for x in range(df.shape[1]):
          plt.text(x , y, '%.4f' % df.iloc[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             zorder=10000
          )
    plt.show()

def plot_dataframe_with_separate_diag_color(df, title=""):
    """The function `plot_dataframe_with_separate_diag_color` creates a plot of a DataFrame with diagonal
    elements in one color and off-diagonal elements in another color.
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe representation of the confusion matrix, with class labels as index and columns
    title : str, optional
        a string that represents the title of the plot., by default ""
    """
    __, ax = plt.subplots(figsize=1.5*np.array(df.shape))
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
    """Plot the recall matrix (Producer accuracy) from the confusion matrix."""
    plot_dataframe_with_separate_diag_color(
        (df_matrix.T/df_matrix.sum(axis=1)).T,
        "recall matrix"
        )
    plt.show()

def precision_matrix_with_sep_diag(df_matrix):
    """Plot the precision matrix (User accuracy) from the confusion matrix."""
    plot_dataframe_with_separate_diag_color(
        df_matrix/df_matrix.sum(axis=0),
        "precision matrix"
        )
    plt.show()

def F1_recall_precision_barh(F1, recall, precision, classes, conf_mat):
    """Make an horizontal bar plot for each of the metric, for each class

    Parameters
    ----------
    F1 : list or np.array
        List of the F1-scores of each class.
    recall : list or np.array
        List of the recall (Producer accuracy) of each class.
    precision : list or np.array
        List of the precision (User accuracy) of each class.
    classes : list or np.array
        List of the class codes or labels.
    conf_mat : np.array
        A numpy representation of the confusion matrix.
    """
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
    """Make an horizontal bar plot for each of the metric, for each class, and compares the score of a method with score of a baseline.

    Parameters
    ----------
    F1 : list or np.array
        List of the F1-scores of each class for the compared method.
    recall : list or np.array
        List of the recall (Producer accuracy) of each class for the compared method.
    precision : list or np.array
        List of the precision (User accuracy) of each class for the compared method.
    F1_baseline : list or np.array
        List of the F1-scores of each class for the baseline method.
    recall_baseline : list or np.array
        List of the recall (Producer accuracy) of each class for the baseline method.
    precision_baseline : list or np.array
        List of the precision (User accuracy) of each class for the baseline method.
    classes : list or np.array
        List of the class codes or labels for both methods.
    conf_mat : np.array
        A numpy representation of the confusion matrix for the compared method.
    """
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

def draw_metrics(path, rows, baseline_row=None, savepath=None):
    """Reads given rows of a given excel file, and plot the obtained results.

    Parameters
    ----------
    path : str
        The path to the excel file. The format of the file must be similar to the output of "land_use_classification_pipeline.py".
    rows : array_like or int
        List of the rows to plot.
    baseline_row : int, optional
        If provided, the results of each row will be compared to this baseline, by default None
    savepath : str, optional
        The path to save the image, by default None = the image is not saved.
    """
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


def sources_importance(df, rows, sources_names, savepath=None):
    """Plot for each source the F1-score obtained when trained and evaluated using only this source.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe corresponding to the opened excel file with the results.
    rows : array_like
        List of the rows to plot. Each row correspond in the excel file to the results of one source.
    sources_names : array_like
        List of the names of the sources.
    savepath : str, optional
        Path to save the output image, by default None

    Example
    -------

    sources_importance(
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
            color=colors,
            height=0.5
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
        plt.text(score, i, f" {score :.2f}", ha='left', va='center')
    plt.ylabel("Sources")
    plt.xlabel("mF1")
    plt.xlim(0, 1)
    plt.ylim(-0.5, len(rows)-0.5)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
    plt.show()

def loco(df, rows, sources_names, baserow, savepath=None, xlims=None):
    """ "Leave One Covariate Out".

    Plot for each source the F1-score lost 
    when trained and evaluated using all sources except this source.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe corresponding to the opened excel file with the results.
    rows : array_like
        List of the rows to plot. Each row correspond in the excel file to the results of one source.
    sources_names : array_like
        List of the names of the sources.
    baserow : int
        The row in the excel files corresponding to the model trained and evaluated using all sources.
    savepath : str, optional
        Path to save the output image, by default None
    xlims : list, optional
        x limits of the plot, as in matplotlib, by default None

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
    plt.barh(sources_names, selected_metrics, color=COLOR_F1, height=0.5)
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


def mixedTrainSet(df, train_dep, test_dep):
    """Plot function for mixed train set experience.

    The results are shown in the form of a latex table and of a 3D plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe corresponding to the opened excel file with the results.
        It is expected that the comment column contains formated information about
        the mixing of the two departements.
    train_dep: str
        Train_dep is the name of the training department for the mixed training set.
    test_dep: str
        Name of the Test department for evaluation of mF1
    
    """

    #Extract the roxs with mixed trained set
    # and construct a matrix of the proportion of each study area
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

    # Create the 3D plot

    X, Y = result_table.columns.values, result_table.index.values
    X, Y = np.meshgrid(X, Y)
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
    sources_importance(df,
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
    sources_importance(df,
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
    sources_importance(df,
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


