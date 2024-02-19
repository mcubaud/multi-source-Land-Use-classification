#%% -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:05:33 2022

@author: MCubaud
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

path_style = os.path.join(
    "C:\\",
    "Users",
    "mcubaud",
    "Documents",
    "Donnees", "OCS_GE", "IA_2019",
    "OCS_GE_1-1_2019_SHP_LAMB93_D032_2022-06-21",
    "OCS_GE", "3_SUPPLEMENTS_LIVRAISON_2022-06-00236",
    "NomenclatureOCSGE.csv"
    )

nomenclature = pd.read_csv(path_style, sep="	", index_col="CODE")

def color_from_US_or_CS(US_or_CS):
    """Return the color associated to the given US
        or CS classes according to the nomenclatura"""
    colors = nomenclature.loc[US_or_CS, ["R", "V", "B"]]
    return np.array(colors)/255

def create_legend():
    CODE = nomenclature.index
    US = CODE[CODE.str.contains("US")]
    labels = nomenclature.loc[US, "LIBELLE_EN"]
    colors = color_from_US_or_CS(US)
    def create_custom_legend(colors, labels):
        legend_elements = []

        for color, label in zip(colors, labels):
            legend_elements.append(Patch(facecolor=color, edgecolor="black", label=label))

        return legend_elements
    legend_elements = create_custom_legend(colors, labels)

    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off the axis

    legend = ax.legend(handles=legend_elements, loc='upper left')
    for legend_handle in legend.legendHandles:
        legend_handle.set_height(10)  # Set the desired height for the legend elements
        legend_handle.set_width(10)   # Set the desired width for the legend elements
    frame = legend.get_frame()
    frame.set_facecolor('white')  # You can set the background color of the legend

    plt.show()
# %%
