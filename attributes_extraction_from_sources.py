# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import path
import numpy as np
# import shapefile
import pandas as pd
import geopandas as gpd

import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from shapely import geometry
from shapely import errors
from shapely.validation import make_valid
from shapely.wkt import loads, dumps
from shapely.geometry import box

import rasterio
import rasterio.mask
from rasterio.plot import show
from osgeo import gdal

from sklearn.preprocessing import OneHotEncoder

from AdjacenceMatrix import open_ocsge_adj_matrix

from scipy.stats import pearsonr



def add_convexity(geodataframe):
    r"""
    Add to each polygon a geometrical convexity index.

    Convexity is here defined as :

    .. math::
       \frac{ area(polygon) }{ area(convex\_hull(polygon)) }

    It represents its convexity in term of area,
    a convexity of 1 being a real convex polygon, and smaller convexities
    are polygons more or less concave.
    """
    geodataframe.loc[
        :, "convexite"] = geodataframe.area / geodataframe.convex_hull.area

def add_concavity(geodataframe):
    r"""
    Add to each polygon a geometrical concavity index.

    Concavity is here defined as :

    .. math::
       \frac{ perimeter(convex\_hull(polygon)) }{ perimeter(polygon) }

    It represents its convexity in term of perimeter,
    a convexity of 1 being a real convex polygon, and smaller convexities
    are polygons more or less concave.
    """
    geodataframe.loc[:, "concavite"] = \
        geodataframe.convex_hull.length / geodataframe.length

def add_compacity(geodataframe):
    r"""
    Add to each polygon Miller's geometrical compacity index.

    Compacity is here defined as :

    .. math::
       \frac{ 4 \times \pi \times area(polygon) }{ perimeter(polygon)^2 }
    """
    geodataframe.loc[:, "compacite"] = \
        4 * np.pi * geodataframe.area / geodataframe.length**2

def add_elongation(geodataframe):
    r"""
    Add to each polygon its
    """
    def elongation(geom):
        mbbox = geom.minimum_rotated_rectangle
        x, y = mbbox.exterior.xy
        l1 = ((x[0] - x[1])**2 + (y[0] - y[1])**2)**0.5
        l2 = ((x[2] - x[1])**2 + (y[2] - y[1])**2)**0.5
        return max(l1, l2)/min(l1, l2)

    geodataframe.loc[
        :, "elongation"] = geodataframe.geometry.apply(lambda geom: elongation(geom))


def turning_function(poly):
    #coordonnées du polygone
    x, y = poly.exterior.coords.xy
    x = x[:-1];y = y[:-1]
    perimeter = poly.exterior.length
    dx = np.concatenate((x[1:],[x[0]])) - x
    dy = np.concatenate((y[1:],[y[0]])) - y
    thetas = np.arctan2(dy, dx)
    norm_curv_abs = np.cumsum(np.sqrt(dx**2+dy**2))/perimeter

    #Transformation en fonction à palliers :
    thetas2 = [thetas[i//2] for i in range(2*len(thetas))]
    norm_curv_abs2 = [0] + [
        norm_curv_abs[i//2] for i in range(2*len(norm_curv_abs)-1)]

    plt.figure()
    plt.plot(norm_curv_abs2, thetas2)
    plt.show()

    return thetas

def plot_turning(x, y, thetas):
    f=plt.figure(); ax = f.add_subplot(1,1,1);
    plt.scatter(x, y);
    a = 2.5
    for i in range(len(x)):
        s = np.sign(-abs(thetas[i])+np.pi/2)
        plt.annotate(str(i), (x[i], y[i]+2*a))
        plt.plot([x[i], x[i]+a*s],[y[i], y[i] + a*s*np.tan(thetas[i])],"--k")
        plt.plot([x[i], x[i]+a],[y[i], y[i]],"--k")
        ax.add_patch(Arc([x[i], y[i]], 2*a, 2*a, theta1=0, theta2=180/np.pi*thetas[i]))


def polygon_signature(poly, nb_points=20):
    x, y = np.array(poly.exterior.coords.xy)
    x0, y0 = poly.centroid.coords.xy
    dx = np.concatenate((x[1:],[x[0]])) - x
    dy = np.concatenate((y[1:],[y[0]])) - y
    perimeter = poly.exterior.length
    norm_curv_abs = np.cumsum(
        np.concatenate(([0],np.sqrt(dx**2+dy**2)[:-1]))
        )/perimeter
    s = [i/nb_points for i in range(nb_points)]
    dist = np.array(
        [i - norm_curv_abs for i in s])
    w = (dist>=0)
    closest_pts = np.array([
        np.arange(len(x))[w[i]][
            np.argmin([dist[i][w[i]]])]
        for i in range(nb_points)])
    b = np.array([dist[i, closest_pts[i]] for i in range(nb_points)])
    dc = norm_curv_abs[1:]-norm_curv_abs[:-1]
    x_sampl = x[closest_pts] \
        + b/dc[closest_pts] * (
            dx[closest_pts]
            )
    y_sampl = y[closest_pts] \
        + b/dc[closest_pts] * (
            dy[closest_pts]
            )
    # plt.figure()
    # plt.plot(x, y); plt.plot(x_sampl, y_sampl, "o")

    signature = np.sqrt((x_sampl-x0)**2 + (y_sampl-y0)**2)

    #normalisation : on divise par la norme L2
    signature = signature/np.sqrt(
        np.sum(
            signature**2)
        )

    #On fait commencer par le point le plus proche
    imin = np.argmin(signature)

    signature = np.concatenate(
        (
            signature[imin:],
            signature[:imin]
            )
        )

    # plt.figure()
    # plt.plot(signature)
    # plt.show()
    return signature

def recuperer_oso(poly):
    with rasterio.open(
        path_oso) as src:
            out_image, out_transform = rasterio.mask.mask(src,
                                                          [poly],
                                                          crop=True,
                                                          pad=True,
                                                          all_touched=True)
            if len(out_image[out_image!=0])>0:
                mode = np.unique(
                    out_image[out_image!=0]
                    )[np.argmax(
                        np.unique(
                            out_image[out_image!=0], return_counts = True)[1]
                        )]
                return mode
            return 0

def folder_of_raster_file_to_extent_gpkg(input_folder, output_gpkg):
    # Create an empty GeoDataFrame to store the extents and paths
    gdf = gpd.GeoDataFrame(columns=["path"], geometry=[])

    # Iterate over the raster files in the input folder
    for file_name in os.listdir(input_folder):
        try:
            file_path = os.path.join(input_folder, file_name)

            # Open the raster file and get its extent
            with rasterio.open(file_path) as src:
                bounds = src.bounds

            # Create a geometry from the extent
            geom = box(*bounds)

            # Add the extent and path to the GeoDataFrame
            gdf = gdf.append(
                {"path": file_path, "geometry": geom},
                ignore_index=True)
        except Exception as e:
            print(e)

    # Save the GeoDataFrame to the output GeoPackage file
    gdf.to_file(output_gpkg, driver="GPKG")

def recuperer_raster_1D_dalle(poly, dalles, cols):
    #poly doit être un geodataframe
    poly_serie = gpd.GeoSeries(poly.geometry)
    # print(poly, type(poly))
    poly_df = gpd.GeoDataFrame(poly_serie,
                            geometry=poly_serie.geometry,
                            crs=dalles.crs)
    # print(poly, type(poly))
    intersection = dalles.overlay(poly_df,
                    how='intersection',
                    keep_geom_type=True)

    path_dalles = intersection["path"].to_list()
    mean = 0
    std = 0
    nb_total_pixels = 0
    #Il y a un problème pénible d'inversion d'axes
    coords = poly.geometry.exterior.coords.xy
    poly2 = geometry.Polygon(
        np.array([coords[0], coords[1]]).T)
    for path_dalle in path_dalles:
        with rasterio.open(
            path_dalle
            ) as src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src,
                                                                  [poly2],
                                                                  crop=True,
                                                                  pad=True,
                                                                  all_touched=True
                                                                  )
                    sumax0 = out_image.sum(axis=0)
                    if len(out_image[out_image!=src.nodata])>0:
                        nb_pixels = len(sumax0[sumax0!=src.nodata])
                        nb_total_pixels += nb_pixels

                        mean += np.sum(out_image[out_image!=src.nodata])
                        std += nb_pixels * np.var(out_image[out_image!=src.nodata])
                except Exception as e:
                    print(f"{poly.ID_1} didn't work")
                    print(e)
                    plt.figure()
                    ax = plt.subplot()
                    show(src, ax=ax)
                    poly_df.plot(ax=ax, color="red")
                    plt.show()
    if nb_total_pixels == 0:
        return pd.Series([0, 0], index=cols)
    else:
        mean = mean / nb_total_pixels
        std = np.sqrt(std / nb_total_pixels)
        return pd.Series([mean, std], index=cols)

def recuperer_Ortho(poly):
    path_dalles = trouver_bonnes_dalles(poly, "BD_ORTHO")
    cols = ["meanR", "meanV", "meanB",
    "stdR", "stdV", "stdB"]
    means = [0, 0, 0]
    stds = [0, 0, 0]
    nb_total_pixels = 0
    #Il y a un problème pénible d'inversion d'axes
    coords = poly.geometry.exterior.coords.xy
    poly2 = geometry.Polygon(
        np.array([coords[0], coords[1]]).T)
    for path_dalle in path_dalles:
        with rasterio.open(
            path_dalle
            ) as src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src,
                                                                  [poly2],
                                                                  crop=True,
                                                                  pad=True,
                                                                  all_touched=True
                                                                  )
                    sumax0 = out_image.sum(axis=0)
                    if len(sumax0[sumax0!=0])>0:
                        nb_pixels = len(sumax0[sumax0!=0])
                        nb_total_pixels += nb_pixels
                        for i in range(3):#RGB
                            means[i] += np.sum(out_image[i, sumax0!=0])/255
                            stds[i] += nb_pixels * np.var(out_image[i, sumax0!=0])/255
                except Exception:
                    print(f"Pas marché {poly.ID_1}")
                    plt.figure()
                    ax = plt.subplot()
                    show(src, ax=ax)
                    poly.plot(ax=ax, color="red")
                    plt.show()
    if nb_total_pixels == 0:
        return pd.Series([0]*6, index=cols)
    else:
        means = np.array(means)/nb_total_pixels
        stds = np.sqrt(np.array(stds)/nb_total_pixels)
        return pd.Series(np.concatenate((means, stds)), index=cols)

def recuperer_RVBI(poly, opened_dalles=None):
    path_dalles = trouver_bonnes_dalles(poly,
                                      "RVBI",
                                      opened_dalles
                                      )
    cols = ["meanR", "meanV", "meanB", "meanPIR",
            "stdPIR", "stdR", "stdV", "stdB"]
    means = [0, 0, 0, 0]
    stds = [0, 0, 0, 0]
    nb_total_pixels = 0
    #Il y a un problème pénible d'inversion d'axes
    coords = poly.geometry.exterior.coords.xy
    poly2 = geometry.Polygon(
        np.array([coords[0], coords[1]]).T)
    for path_dalle in path_dalles:
        with rasterio.open(
            path_dalle
            ) as src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src,
                                                                  [poly2],
                                                                  crop=True,
                                                                  pad=True,
                                                                  all_touched=True)
                    sumax0 = out_image.sum(axis=0)
                    if len(sumax0[sumax0!=0])>0:
                        nb_pixels = len(sumax0[sumax0!=0])
                        nb_total_pixels += nb_pixels
                        for i in range(4):#PIRRGB
                            means[i] += np.sum(out_image[i, sumax0!=0])/255
                            stds[i] += nb_pixels * np.var(out_image[i, sumax0!=0])/255
                except Exception:
                    print(f"Pas marché {poly.ID_1}")
                    plt.figure()
                    ax = plt.subplot()
                    show(src, ax=ax)
                    poly.plot(ax=ax, color="red")
                    plt.show()
    if nb_total_pixels == 0:
        return pd.Series([0]*8, index=cols)
    else:
        means = np.array(means)/nb_total_pixels
        stds = np.sqrt(np.array(stds)/nb_total_pixels)
        return pd.Series(np.concatenate((means, stds)), index=cols)

def open_shapefile_dalle():
    basepath = os.path.join(
        "D:\\", "BD_ORTHO",
        "ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D032_2019-01-01",
        "ORTHOHR"
        )
    data_folder = "1_DONNEES_LIVRAISON_2020-05-00114"
    supp_folder = "3_SUPPLEMENTS_LIVRAISON_2020-05-00114"
    sub_folder = "OHR_RVB_0M20_JP2-E080_LAMB93_D32-2019"
    dalles = gpd.read_file(
        os.path.join(
            basepath, supp_folder, sub_folder,
            "dalles.shp"
            )
        )
    return dalles

def suppress_accents(string):
    if type(string)!=str:
        return string
    string = string.lower()
    liste_forbiden_caracters = [
        "-","'","à","â","é","è","ê","ë","î","ï","ô","ù","û","ü","ÿ"
        ]
    liste_replacer = [
        " "," ","a","a","e","e","e","e","i","i","o","u","u","u","y"
        ]
    for i, car in enumerate(liste_forbiden_caracters):
        string = string.replace(car, liste_replacer[i])
    return string


def list_from_text(string):
    try:
        return eval("[" + string + "]")
    except:#sometimes excel strangely delete the first '
        return eval("['" + string + "]")


def classify_osm(osm, classif_table_path, save_folder, condition_code_col):
    """
    Map some of the OpenStreetMap objects to a LU class.

    It can work for nodes, ways and areas.
    The mapping is described in a csv file.
    Parameters
    ----------
    osm : GeoDataFrame
        Geodataframe containing the OSM objects.
    classif_table_path : str
        Path to the csv table.
        The columns of the table should be at least :
            key, values, condition_code_col, CODE_US
    save_folder : str
        Path where the objects are saved.
    condition_code_col : str
        Name of the column with a supplementary condition on other tags.
        Elements of this column must be empty or functions of an osm object
        returning True if the element pass this condition and False otherwise.

    Returns
    -------
    None.

    """

    table = pd.read_csv(classif_table_path)
    grouped_by_LU = table.groupby(by="CODE US")
    grouped_by_keys = table.groupby(by="key")
    os.makedirs(save_folder, exist_ok=True)
    for LU in grouped_by_LU.groups:
        boolean_index = np.zeros(len(osm), dtype=bool)
        this_LU = grouped_by_LU.get_group(LU).set_index("key")
        for key in this_LU.index:

            values = this_LU.loc[key, "values"]

            if key == "*":
                #We should search for any key with these values
                values = list_from_text(values)
                should_negate = False
            else:
                if values == 'any':
                    if key in osm.columns:
                        new_boolean_index = ~osm.loc[:, key].isnull()
                    else:
                        if "," in key:
                            #key is actually a list of keys
                            keys = eval("[" + key + "]")
                            new_boolean_index = osm["other_tags"].apply(
                                lambda x: (
                                    x is not None and (
                                            np.any([
                                                f'"{a_key}"=>' in x for a_key in keys])
                                            )
                                        )
                                    )
                        else:
                            new_boolean_index = osm["other_tags"].apply(
                                lambda x: (
                                    x is not None and (f'"{key}"=>' in x)
                                    )
                                )
                    should_negate = False
                else:
                    if values == 'all others':

                        values = []
                        for list_values in grouped_by_keys.get_group(key)["values"]:
                            if list_values not in ['any', 'all others']:
                                values += list_from_text(list_values)
                        values.append(None)
                        should_negate = True

                    else:#Its a list of values then
                        values = list_from_text(values)
                        should_negate = False

                    if key in osm.columns:
                        new_boolean_index = osm.loc[:, key].isin(values)
                    else:
                        new_boolean_index = osm["other_tags"].apply(
                            lambda x: (
                                x is not None and (
                                        np.any([
                                            f'"{key}"=>"{value}"' in x for value in values])
                                        )
                                    )
                                )
                        if should_negate:#then we have to add when key is not in other_tags
                            new_boolean_index = (
                                new_boolean_index |
                                osm["other_tags"].apply(
                                    lambda x: (
                                        x is None or
                                        x is not None and (
                                            f'"{key}"=>' not in x
                                                )
                                            )
                                        )
                                )


            if ~this_LU.isna().loc[key, condition_code_col]:
                str_code_condition = this_LU.loc[key, condition_code_col]
                #condition is describe in the table as a function of an osm element
                condition_function = eval(str_code_condition)
                condition = osm.apply(condition_function, axis=1)
            else:
                condition = np.ones(len(osm), dtype=bool)

            boolean_index = boolean_index | ((~new_boolean_index if should_negate else new_boolean_index) & condition)
            print(np.sum(boolean_index), "/",len(boolean_index))
        osm_this_LU = osm[boolean_index]
        osm_this_LU.to_file(
            os.path.join(
                save_folder,
                LU + ".gpkg")
            )

def filtrer_foursquare(fsq, BD_TOPO, communes):

    fsq["id"] = np.arange(len(fsq))

    #filtre sur les infos de catégories
    fsq = fsq[~fsq["cat_id"].isna()]

    #filtre sur la position dans un bâtiment
    fsq = fsq.to_crs(BD_TOPO.crs)
    fsq = fsq.overlay(BD_TOPO.loc[:, ["NATURE", "geometry"]],
                      how="intersection",
                      keep_geom_type=True)

    #filtre sur ce qu'il y a dans les églises
    fsq =  fsq[
        (fsq["NATURE"].isin(
            ["Eglise", "Chapelle"]) &\
        fsq["cat_id"].isin(
            [12101, 16000, 16020, 16025, 16026])
        ) | ~ fsq["NATURE"].isin(["Eglise", "Chapelle"])]


    #filtre sur ce qu'il y a dans les mairies et préfectures

    za = gpd.read_file(path_za_bd_topo)

    pts_dans_za = fsq_filtered.overlay(za,
                      how="intersection",
                      keep_geom_type=True)

    pts_dans_mairie = pts_dans_za[
        pts_dans_za["NATURE_2"].isin([
            "Mairie", "Préfecture", "Sous-préfecture"
            ])
        ]

    fsq = fsq[
        (fsq.id.isin(pts_dans_mairie.id) &\
         (fsq["cat_id"]//1000)==12
         ) | ~fsq.id.isin(pts_dans_mairie.id)]


    #filtre du point dans la bonne commune
    fsq = fsq.overlay(communes.loc[:,["NOM", "geometry"]],
                      keep_geom_type=True)
    fsq = fsq[
        fsq["NOM"].apply(suppress_accents)==fsq["locality"].apply(suppress_accents)]

    #filtre pas trop de points au même endroit
    buffered = fsq.copy()
    buffered.geometry = fsq.buffer(1)
    intersection = fsq.overlay(buffered,
                               keep_geom_type=True)
    uniques, counts = np.unique(
        intersection["id_2"],
        return_counts=True)
    fsq = fsq[counts<3]
    return fsq

def stack_RVBI():
    if departement == 32:
        basepath_RVB = os.path.join(
            "D:\\", "BD_ORTHO",
            "ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D032_2019-01-01",
            "ORTHOHR"
            )
        data_folder_RVB = "1_DONNEES_LIVRAISON_2020-05-00114"
        sub_folder_RVB = "OHR_RVB_0M20_JP2-E080_LAMB93_D32-2019"
        basepath_IRC = os.path.join(
            "D:\\", "IRC",
            "ORTHOHR_1-0_IRC-0M20_JP2-E080_LAMB93_D032_2019-01-01",
            "ORTHOHR"
            )
        data_folder_IRC = "1_DONNEES_LIVRAISON_2020-08-00106"
        sub_folder_IRC = "OHR_IRC_0M20_JP2-E080_LAMB93_D32-2019"

        output_folder = os.path.join(
            basepath_RVB,
            data_folder_RVB,
            "OHR_RVBI_0M20_JP2-E080_LAMB93_D32-2019"
            )

    elif departement == 69:

        basepath_RVB = os.path.join(
            "E:\\", "BD_ORTHO_69",
            "ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D069_2020-01-01",
            "ORTHOHR"
            )
        data_folder_RVB = "1_DONNEES_LIVRAISON_2021-03-00176"
        sub_folder_RVB = "OHR_RVB_0M20_JP2-E080_LAMB93_D69-2020"
        basepath_IRC = os.path.join(
            "E:\\", "BD_ORTHO_69",
            "ORTHOHR_1-0_IRC-0M20_JP2-E080_LAMB93_D069_2020-01-01",
            "ORTHOHR"
            )
        data_folder_IRC = "1_DONNEES_LIVRAISON_2021-03-00246"
        sub_folder_IRC = "OHR_IRC_0M20_JP2-E080_LAMB93_D69-2020"

        output_folder=os.path.join(
            basepath_IRC,
            data_folder_IRC,
            "OHR_RVBI_0M20_JP2-E080_LAMB93_D69-2020"
            )
    os.makedirs(output_folder, exist_ok=True)

    try:
        already_done = list(np.genfromtxt(os.path.join(basepath_RVB,
                                                  data_folder_RVB,
                                                  "Done.txt"),
                                     dtype=str))
    except:
        already_done = []
    for path_RVB in path.Path(
            os.path.join(
                basepath_RVB,
                data_folder_RVB,
                sub_folder_RVB)
            ).walkfiles():
        if path_RVB.ext == ".jp2":
            nom = path_RVB.split(os.sep)[-1]
            print(nom)
            if not nom in already_done:
                path_IRC = os.path.join(
                    basepath_IRC,
                    data_folder_IRC,
                    sub_folder_IRC,
                    nom.replace("E080", "IRC-E080")
                    )

                # Read metadata of first file
                with rasterio.open(path_RVB) as src0:
                    meta = src0.meta

                # Update meta to reflect the number of layers
                meta.update(count = 4)

                # Read each layer and write it to stack
                with rasterio.open(
                        os.path.join(
                            output_folder,
                            nom
                            ), 'w', **meta) as dst:
                    with rasterio.open(path_IRC) as src2:
                        dst.write_band(1, src2.read(1))
                    with rasterio.open(path_RVB) as src1:
                        for i in range(1,4):
                            dst.write_band(i+1, src1.read(i))
                already_done.append(nom)
                np.savetxt(os.path.join(basepath_RVB,
                                        data_folder_RVB,
                                        "Done.txt"),
                           already_done,
                           "%100s"
                           )





def recuperer_IRC(poly):
    path_dalles = trouver_bonnes_dalles(poly, "IRC")
    cols = ["meanPIR", "stdPIR"]
    mean = 0
    std = 0
    nb_total_pixels = 0
    #Il y a un problème pénible d'inversion d'axes
    coords = poly.geometry.exterior.coords.xy
    poly2 = geometry.Polygon(
        np.array([coords[0], coords[1]]).T)
    for path_dalle in path_dalles:
        with rasterio.open(
            path_dalle
            ) as src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src,
                                                                  [poly2],
                                                                  crop=True,
                                                                  pad=True,
                                                                  all_touched=True)
                    #On ne veut que la bande PIR, les autres étant déjà dans la RVB
                    out_image = out_image[0]
                    if len(out_image[out_image!=0])>0:
                        nb_pixels = len(out_image[out_image!=0])
                        nb_total_pixels += nb_pixels
                        mean += np.sum(out_image[out_image!=0])/255
                        std += nb_pixels * np.var(out_image[out_image!=0])/255
                except Exception:
                    print(f"Pas marché {poly.ID_1}")
                    plt.figure()
                    ax = plt.subplot()
                    show(src, ax=ax)
                    poly.plot(ax=ax, color="red")
                    plt.show()
    if nb_total_pixels == 0:
        return pd.Series( [0, 0], index=cols)
    else:
        mean = mean/nb_total_pixels
        std = np.sqrt(std/nb_total_pixels)
        return  pd.Series( [mean, std], index=cols)



def trouver_bonnes_dalles(poly, image, opened_dalles=None):
    """


    Parameters
    ----------
    poly : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # print(poly, type(poly))
    path_orthos_departements = pd.read_csv(
        os.path.join(
            "E:",
            "code_path_orthos_departements.csv"
            ),
        index_col=0
        )
    basepath = data_folder = supp_folder = sub_folder = ""
    try:
        #set the variables basepath, data_folder, supp_folder and sub_folder
        basepath, data_folder, supp_folder, sub_folder = eval(path_orthos_departements.loc[departement, image])
    except:
        print("Bad departement or unknown image or code error")

    # if image == "BD_ORTHO":
    #     basepath = os.path.join(
    #         "D:\\", "BD_ORTHO",
    #         "ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D032_2019-01-01",
    #         "ORTHOHR"
    #         )
    #     data_folder = "1_DONNEES_LIVRAISON_2020-05-00114"
    #     supp_folder = "3_SUPPLEMENTS_LIVRAISON_2020-05-00114"
    #     sub_folder = "OHR_RVB_0M20_JP2-E080_LAMB93_D32-2019"
    # elif image == "RVBI":
    #         basepath = os.path.join(
    #             "D:\\", "BD_ORTHO",
    #             "ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D032_2019-01-01",
    #             "ORTHOHR"
    #             )
    #         data_folder = "1_DONNEES_LIVRAISON_2020-05-00114"
    #         supp_folder = "3_SUPPLEMENTS_LIVRAISON_2020-05-00114"
    #         sub_folder = "OHR_RVBI_0M20_JP2-E080_LAMB93_D32-2019"
    # elif image == "IRC":
    #     basepath = os.path.join(
    #         "D:\\", "IRC",
    #         "ORTHOHR_1-0_IRC-0M20_JP2-E080_LAMB93_D032_2019-01-01",
    #         "ORTHOHR"
    #         )
    #     data_folder = "1_DONNEES_LIVRAISON_2020-08-00106"
    #     supp_folder = "3_SUPPLEMENTS_LIVRAISON_2020-08-00106"
    #     sub_folder = "OHR_IRC_0M20_JP2-E080_LAMB93_D32-2019"
    # else:
    #     print("Unknown image")

    if opened_dalles is None:
        dalles = gpd.read_file(
            os.path.join(
                basepath, supp_folder, sub_folder,
                "dalles.shp"
                )
            )
        opened_dalles = dalles
    else:
        dalles = opened_dalles

    #poly doit être un geodataframe
    poly = gpd.GeoSeries(poly.geometry)
    # print(poly, type(poly))
    poly = gpd.GeoDataFrame(poly,
                            geometry=poly.geometry,
                            crs=dalles.crs)
    # print(poly, type(poly))
    intersection = dalles.overlay(poly,
                    how='intersection',
                    keep_geom_type=True)

    # dalles_select = dalles[dalles["NOM"].isin(intersection["NOM"])]
    # plt.figure()
    # ax = plt.subplot()
    # dalles_select.plot(ax=ax)
    # # plt.annotate(dalles_select.NOM,
    # #              np.array([
    # #                  dalles_select.geometry.centroid.x,
    # #                  dalles_select.geometry.centroid.y
    # #                  ])
    # #          )
    # poly.plot(ax=ax, color="red")
    # plt.show()
    bonnes_dalles = intersection.NOM.apply(
        lambda x: os.path.join(
            basepath,
            data_folder,
            sub_folder,
            x.replace("./", "").replace("-IRC",""))
        ).to_list()
    return bonnes_dalles

def build_vrt_from_folder(destName, folder_path):
    src = os.listdir(folder_path)
    src = [os.path.join(folder_path, src_i) for src_i in src]
    vrt = gdal.BuildVRT(destName, src)

def add_points(couche, couche_pts, col_name):
    couche_pts = couche_pts.drop(columns=couche_pts.columns[:-1])
    couche_pts[col_name] = 1

    join = couche.sjoin(
        couche_pts,
        how='inner',
        predicate="intersects"
        )

    aggfunc = {col_name: "count"}

    couche_and_pts = join.dissolve(
        by=couche.index.name,
        aggfunc=aggfunc
        )

    couche[col_name] = couche_and_pts[col_name]
    couche[col_name] = couche[col_name].fillna(0)

    return couche

def mean_neighbours_values(couche, quanti_cols, quali_cols):
    adj, uniques, inv = open_ocsge_adj_matrix(with_diag=False,
                                              path=path_adj_matrix)
    couche = couche.set_index("ID_1")
    df_uniques = pd.DataFrame(inv, index=uniques)

    #L'ordre dans la couche n'est pas forcément
    #le même que dans la matrice d'adjacence
    couche_meme_ordre, __ = couche.align(
        df_uniques, join='right', axis=0)

    #Colones quantitatives : valeurs moyennes
    values = np.array(couche_meme_ordre[quanti_cols])
    values_mean = adj @ values
    new_names = [name+'_mean_1m' for name in quanti_cols]
    couche_meme_ordre[new_names] = values_mean

    #Colones qualitatives : valeurs majoritaires
    if len(quali_cols)>0:
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_cols = OH_encoder.fit_transform(
            couche_meme_ordre[quali_cols].to_numpy())
        encoded_cols_mean = adj @ encoded_cols
        start = 0
        for i, col in enumerate(quali_cols):
            uniques_val = OH_encoder.categories_[i]
            nb_values = len(uniques_val)
            majo = np.argmax(encoded_cols_mean[:, start:start+nb_values], axis=1)
            couche_meme_ordre[col+'_mean_1m'] = uniques_val[majo]
            start += nb_values

    return couche_meme_ordre

def weighted_mean(join):
    def f(x):
        return np.average(x, weights=join.loc[x.index].area)
    return f

def weighted_majority(join):
    def f(x):
        unique = x.unique()
        unique_table = np.array([x==u for u in unique])
        weights = join.loc[x.index].area
        somme = unique_table.dot(weights)
        return unique[np.argmax(somme)]
    return f


def add_vector_layer(ocsge,
                     vector_layer,
                     aggfunc,
                     savepath,
                     layername,
                     how='intersection',
                     new_cols=None):
    """
    Add data from a vector layer to the OCS-GE polygons.

    First the intersection of the vector layer with OCS-GE polygons is made,
    then all the pieces that come from the same OCS-GE polygon are grouped together,
    and the vector layer attributes are aggregated.

    Parameters
    ----------
    ocsge : GeoDataFrame
        The OCSGE.
        It must have a "ID_1" column which is not set as index
    vector_layer : GeoDataFrame
        The vector layer.
    aggfunc : dict
        Dictionary with as keys the name of keeped vector layer attributes and as values aggregation function.
    savepath : str or None
        The path where to save the result. If None, it will not be saved.
    layername : str
        The name of the layer.
    how : str
        Method of spatial overlay: 'intersection', 'union', 'identity', 'symmetric_difference' or 'difference'.
        Use 'union' except if all OCSGE polygons are intersected by an object from the vector layer, then prefer use 'intersection'.
    new_cols : dict or None
        Dictionary describing attributes created at the level of join objects.
        Dictionary keys are the name of the new attributes.
        Dictionary values are functions applied to join.

    Returns
    -------
    ocsge_plus_vector_layer : GeoDataFrame
        OCSGE polygons with former attributes and those added from the vector layer.
    """
    try:
        join = ocsge.overlay(vector_layer,
                             how="intersection",
                             keep_geom_type=(
                                 #if both layer have the same geometric type
                                 ocsge.geom_type.iloc[0]
                                 ==
                                 vector_layer.geom_type.iloc[0]
                                 )
                             )
                              # keep_geom_type=True)
    except errors.TopologicalError:
        print(f"{layername} n'a pas pu être intersecté, tentative de simplifier les coordonnées")
        ocsge.geometry = [
            loads(dumps(geom, rounding_precision=3)) for geom in ocsge.geometry]
        vector_layer.geometry = [
            loads(dumps(geom, rounding_precision=3)) for geom in vector_layer.geometry]
        join = ocsge.overlay(vector_layer,
                             how="intersection",
                             keep_geom_type=(
                                 #if both layer have the same geometric type
                                 ocsge.geom_type.iloc[0]
                                 ==
                                 vector_layer.geom_type.iloc[0]
                                 )
                             )

    ocsge = ocsge.set_index("ID_1")

    if new_cols is not None:
        for col_name in new_cols:
            join[col_name] = new_cols[col_name](join)

    aggfunc = modify_aggfunc(aggfunc, join)

    aggregated = join.dissolve(
        by="ID_1",
        aggfunc=aggfunc
        )

    ocsge[ list(aggfunc.keys()) ] = aggregated[ list(aggfunc.keys()) ]

    if savepath:
        ocsge.to_file(
            savepath,
            layer=layername,
            driver="GPKG")

    ocsge = ocsge.reset_index()
    return ocsge

def surf_if_condition(join, col):
    return ~join[col].isna() * join.area

def frac_surf_if_condition(join, col):
    return ~join[col].isna()

def length_if_condition(join, col):
    return  ~join[col].isna() * join.length

def modify_aggfunc(aggfunc, join):
    if type(aggfunc) == dict:
        for key in aggfunc:
            if aggfunc[key] in [
                    weighted_majority,
                    weighted_mean
                    ]:
                aggfunc[key] = aggfunc[key](join)
    return aggfunc

def positional_encoding(ocsge, d):
    omegas = 1/10000**(2*np.arange(int(d/4))/d).reshape(-1,1)
    #We convert to latitudes and longitudes
    geom = ocsge.geometry
    centroid = geom.centroid.to_crs(epsg=4326)
    x = centroid.x.to_numpy().reshape(1,-1)
    y = centroid.y.to_numpy().reshape(1,-1)
    omegas_x = omegas.dot(x)
    omegas_y = omegas.dot(y)
    cos_x = np.cos(omegas_x)
    sin_x = np.sin(omegas_x)
    cos_y = np.cos(omegas_y)
    sin_y = np.sin(omegas_y)
    p_xy = np.vstack((cos_x, sin_x, cos_y, sin_y))
    return p_xy

def link_values_class(ocsge,
                      quanti_cols,
                      quali_cols,
                      quanti_cols_units,
                      target_col="CODE_US",
                      english=False):
    all_texts = ""
    y = ocsge[target_col]
    liste_US = np.unique(y)

    for US in liste_US:
        is_this_US = y==US
        os.makedirs(
            os.path.join(
                "E:", "comparaison", US
                ),
            exist_ok=True
            )
        if english:
            US = US.replace("US", "LU")
        for i, col in enumerate(quanti_cols):
            unit = quanti_cols_units[i]
            r, p = pearsonr(ocsge[col], is_this_US)
            if p <=0.05 and abs(r)>=0.5:
                new_text = f"Correlation between {col} and {US} : (r={r:.2f}, p={p:.2e})"
                print(new_text)
                all_texts += new_text + "/n"


            plt.figure()
            plt.hist(
                [ ocsge[is_this_US][col], ocsge[~is_this_US][col] ],
                weights = [
                    np.ones_like(ocsge[is_this_US][col]) / len(ocsge[is_this_US][col]),
                    np.ones_like(ocsge[~is_this_US][col]) / len(ocsge[~is_this_US][col])
                    ]
            )
            plt.legend([US, "not "+US])
            plt.title(f"Link {col}/{US} (corr={r :.2f}, p={p :.2e})")
            plt.xlabel(f"{col} ({unit})")
            plt.ylabel("frequency")
            plt.savefig(
                os.path.join(
                    "E:", "comparaison", US, f"{col}.png"
                    )
                )
            plt.show()
            min_col, max_col = ocsge[col].min(), ocsge[col].max()
            sort_col = ocsge[col].sort_values()
            is_this_US_min, is_this_US_max = y[ocsge[col].argmin()]==US, y[ocsge[col].argmax()]==US

            #maximal value such that for all v<v_max, ocsge[col]==v => is_this_US==is_this_US_min
            #It is the previous value of the min for which is_this_US!=is_this_US_min
            arg_v_max = (
                ocsge.loc[sort_col.index][col]
                + (max_col - min_col + 1) * (is_this_US.loc[sort_col.index]==is_this_US_min)
                #this is to ensure we take values for which is_this_US!=is_this_US_min
                ).argmin() - 1
            if arg_v_max > 0:
                v_max = ocsge.loc[sort_col.index][col].iloc[arg_v_max]
                proportion = (
                    ocsge[
                        is_this_US==is_this_US_min
                    ][col] < v_max
                    ).sum() / (
                        is_this_US==is_this_US_min
                        ).sum()
                if proportion > 0.5:
                    new_text = (
                        f'{col} <= {v_max} implies US'
                        f'{["!=", "="][is_this_US_min]} {US}'
                        f'(proportion = {proportion} )'
                        )
                    print(new_text)
                    all_texts += new_text + "/n"

            arg_v_min = (
                ocsge.loc[sort_col.index][col]
                - (max_col - min_col + 1) * (is_this_US.loc[sort_col.index]==is_this_US_max)
                ).argmax() + 1
            if arg_v_min < len(ocsge) - 1:
                v_min = ocsge.loc[sort_col.index][col].iloc[arg_v_min]
                proportion = (
                    ocsge[
                        is_this_US==is_this_US_max
                    ][col] > v_min
                    ).sum() / (
                        is_this_US==is_this_US_max
                        ).sum()
                if proportion > 0.5:
                    new_text = (
                        f'{col} >= {v_min} implies US'
                        f'{["!=", "="][is_this_US_max]} {US}'
                        f'(proportion = {proportion})'
                        )
                    print(new_text)
                    all_texts += new_text + "/n"

        for col in quali_cols:
            for value in np.unique(ocsge[col]):
                is_this_value = ocsge[col]==value
                r, p = pearsonr(is_this_value, is_this_US)
                if p <=0.05 and  abs(r)>=0.5:
                    new_text = (f"Correlation between {col}={value} and {US} : (r={r:.2f}, p={p:.2e})")
                    print(new_text)
                    all_texts += new_text + "/n"

                plt.figure()
                plt.hist(
                    [ is_this_value[is_this_US], is_this_value[~is_this_US] ],
                    weights = [
                        np.ones_like(is_this_value[is_this_US]) / len(is_this_value[is_this_US]),
                        np.ones_like(is_this_value[~is_this_US]) / len(is_this_value[~is_this_US])
                        ]
                )
                plt.legend([US, "not "+US])
                plt.title(f"Link ({col}={value})/{US} (corr={r :.2f}, p={p :.2e})")
                plt.xlabel(f"{col}={value}")
                plt.ylabel("frequency")
                plt.savefig(
                    os.path.join(
                        "E:", "comparaison", US, f"{col}={value}.png"
                        )
                    )
                plt.show()
                confusion_matrix = np.array([
                    [
                        (is_this_value[is_this_US]).sum()/is_this_value.sum(),
                        (is_this_value[~is_this_US]).sum()/is_this_value.sum()
                        ],
                    [
                        (~is_this_value[is_this_US]).sum()/(~is_this_value).sum(),
                        (~is_this_value[~is_this_US]).sum()/(~is_this_value).sum()
                        ]

                    ])
                if np.any(confusion_matrix==0):
                    text = ["", "not "]
                    list_i, list_j = np.where(confusion_matrix==0)
                    for k in range(len(list_i)):
                        i, j = list_i[k], list_j[k]
                        new_text = (
                            f"No {col} {text[j]}= {value} & {text[i]}{US}"
                            )
                        print(new_text)
                        all_texts += new_text + "/n"

    return all_texts

def TF_IDF(ocsge, columns_numerical_of_a_source):
    liste_US = np.unique(ocsge["CODE_US"])
    TF_IDF_matrix = pd.DataFrame(index=liste_US, columns=columns_numerical_of_a_source)
    for US in liste_US:
        for col in columns_numerical_of_a_source:
            TF = ocsge[
                ocsge["CODE_US"]==US
                ][col].sum() /\
                ocsge[
                    ocsge["CODE_US"]==US
                    ][columns_numerical_of_a_source].sum().sum()
            IDF = np.log10(
                len(ocsge) / (ocsge[col]!=0).sum()
                )
            TF_IDF_matrix.loc[US, col] = TF * IDF
    return TF_IDF_matrix

#________________________________________________________________________________________________________________________


departement = 32

if departement == 32:

    path_ocsge = os.path.join(
        "Documents",
        "Donnees", "OCS_GE", "IA_2019",
        "OCS_GE_1-1_2019_SHP_LAMB93_D032_2022-06-21",
        "OCS_GE", "1_DONNEES_LIVRAISON_2022-06-00236",
        "OCSGE_1-1_SHP_LAMB93_32-2019",
        "OCCUPATION_SOL.shp"
        )

    path_bd_topo = os.path.join(
        "Documents",
        "Donnees", "BD_TOPO_2022",
        "BDTOPO_3-0_TOUSTHEMES_SHP_LAMB93_D032_2022-06-15",
        "BDTOPO", "1_DONNEES_LIVRAISON_2022-06-00168",
        "BDT_3-0_SHP_LAMB93_D032-ED2022-06-15"
        )

    path_osm = os.path.join(
        "Documents", "Donnees",
        "OSM_22_09_2022",
        "osm_polygons.gpkg"
        )

    path_osm_pts = os.path.join(
        "Documents", "Donnees",
        "OSM_22_09_2022",
        "osm_pts.gpkg"
        )

    path_osm_lines = os.path.join(
        "Documents", "Donnees",
        "OSM_22_09_2022",
        "osm_lines.gpkg"
        )

    savepath_classified_osm =  os.path.join(
        "Documents", "Donnees",
        "OSM_22_09_2022"
        )

    path_oso = os.path.join("Documents",
    "Donnees", "OSO", "OSO_2019.tif")

    path_CLC = os.path.join(
        "Documents",
        "Donnees",
        "CLC",
        "CLC2012_GERS.shp"
        )
    col_CLC = "CODE_12"

    path_iris = os.path.join(
        "Documents", "Donnees", "IRIS", "IRIS_popu_2019.gpkg"
        )

    path_foncier = os.path.join(
        "Documents", "Donnees", "Foncier",
        "Data", "DGFIP", "D032_2017",
        "dgfip_2017.shp"
        )

    path_adj_matrix = ""

    path_carroyage_INSEE = os.path.join(
        "E:\\",
        "Filosofi2017_carreaux_200m_shp",
        "carroyage_INSEE_32.gpkg"
        )

    path_RPG = os.path.join(
        "D:", "RPG", "RPG_2-0_SHP_LAMB93_R76-2019",
        "RPG", "1_DONNEES_LIVRAISON_2019",
        "RPG_2-0_SHP_LAMB93_R76-2019",
        "ILOTS_ANONYMES.shp"
        )

elif departement == 69:

    path_ocsge = os.path.join(
        "D:\\",
        "OCSGE_69",
        "FINAL",
        "OCSGE_69_2020.shp"
        )

    path_bd_topo = os.path.join(
        "D:\\",
        "BD_TOPO_69",
        "BDTOPO_3-0_TOUSTHEMES_SHP_LAMB93_D069_2020-06-15",
        "BDTOPO",
        "1_DONNEES_LIVRAISON_2020-06-00047",
        "BDT_3-0_SHP_LAMB93_D069-ED2020-06-15"
        )

    path_osm = os.path.join(
        "D:\\",
        "OSM_69",
        "osm_polygons_repare.gpkg"
        )

    path_osm_pts = os.path.join(
        "D:\\",
        "OSM_69",
        "osm_pts.gpkg"
        )

    path_osm_lines = os.path.join(
        "D:\\",
        "OSM_69",
        "osm_lines.gpkg"
        )

    savepath_classified_osm =  os.path.join(
        "D:\\",
        "OSM_69"
        )

    path_oso = os.path.join("D:\\",
    "OSO_2020", "DATA", "OCS_2020.tif")

    path_CLC = os.path.join(
        "D:\\",
        "CLC",
        "CLC_2018_69.gpkg"
        )

    col_CLC = "CODE_18"

    path_iris = os.path.join(
        "E:\\", "IRIS", "IRIS_popu_2020_69.gpkg"
        )

    path_foncier = os.path.join(
        "E:\\", "DGFIP", "usages_2021_69.shp"
        )

    path_adj_matrix = os.path.join(
       "E:\\", "Adj_matrix_69"
       )

    path_carroyage_INSEE = os.path.join(
        "E:\\",
        "Filosofi2017_carreaux_200m_shp",
        "carroyage_INSEE_69.gpkg"
        )

    path_RPG = os.path.join(
        "E:",
        "RPG_2-0_SHP_LAMB93_R84_2020",
        "ILOTS_ANONYMES.shp"
        )

path_bati_bd_topo = os.path.join(
    path_bd_topo,
    "BATI",
    "BATIMENT.shp")

path_za_bd_topo = os.path.join(
    path_bd_topo,
    "SERVICES_ET_ACTIVITES",
    "ZONE_D_ACTIVITE_OU_D_INTERET.shp"
    )

path_ERP_bd_topo = os.path.join(
    path_bd_topo,
    "SERVICES_ET_ACTIVITES",
    "ERP.shp"
    )

path_communes_bd_topo = os.path.join(
    path_bd_topo,
    "ADMINISTRATIF",
    "COMMUNE.shp"
    )

path_cimetieres_bd_topo = os.path.join(
    path_bd_topo,
    "BATI",
    "CIMETIERE.shp"
    )

path_aerodrome = os.path.join(
    path_bd_topo,
    "TRANSPORT",
    "AERODROME.shp"
    )

path_routes = os.path.join(
    path_bd_topo,
    "TRANSPORT",
    "TRONCON_DE_ROUTE.shp"
    )

path_train = os.path.join(
    path_bd_topo,
    "TRANSPORT",
    "TRONCON_DE_VOIE_FERREE.shp"
    )

path_river = os.path.join(
    path_bd_topo,
    "HYDROGRAPHIE",
    "SURFACE_HYDROGRAPHIQUE.shp"
    )




#%%Attributs géométriques

# ocsge = shapefile.Reader(path_ocsge)

ocsge = gpd.read_file(path_ocsge)

# #réparation des géométries
# #boucher les trous:
# ocsge["geometry"] = ocsge.geometry.buffer(1e-3).buffer(-1e-3)
# #enlever des petits morceaux qui flottent
# ocsge["geometry"] = ocsge.geometry.buffer(-2e-3).buffer(2e-3)

ocsge["surface"] = ocsge.area
add_convexity(ocsge)
add_compacity(ocsge)
add_elongation(ocsge)

#Nb de trous
ocsge["holes"] = ocsge[
    "geometry"
    ].interiors.apply(
        lambda x: not x is None and len(x)
        )

#Signature
ocsge[
    [f"signature_{i}" for i in range(20)]
    ] = ocsge.geometry.apply(
        lambda x: pd.Series(polygon_signature(x))
        )

#%%Statistiques géométriques des polygones OCSGE

scale = ['log', 'linear', 'linear', 'log', 'linear']
f = plt.figure(figsize=(17,13))
for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
    for j, column in enumerate(['surface', 'convexite',
                                'compacite','elongation',
                                #'concavite'
                                ]):
        ax = plt.subplot(4, 5, 1+5*i+j)
        plt.title(f"{US} {column}")
        plt.yscale(scale[j])
        ocsge[ocsge["CODE_US"]==US].loc[:, column].hist(ax=ax)

#%%
ocsge = gpd.read_file(path_ocsge)

batiments = gpd.read_file(path_bati_bd_topo)#Batiments de la BD TOPO

#Intersection batiments et OCSGE
# join = gpd.sjoin(ocsge, batiments, how="inner", predicate="intersects")
join = ocsge.overlay(batiments, how='union', keep_geom_type=True)

#Enlever les batiments hors de l'emprise de l'OCSGE
join = join.drop(index=(np.where(join.loc[:, "ID_1"].isna())[0]))

aggfunc = {#Comment seront re-agrégées les colonnes
    # "CODE_CS": "first",
    # "CODE_US": "first",
    # "surface": "first",
    # "convexite": "first",
    # "compacite": "first",
    # "elongation": "first",
    "HAUTEUR": "mean"
    }

usages_batiments = [
    'Agricole', 'Annexe', 'Commercial et services',
    'Indifférencié', 'Industriel', 'Religieux',
    'Résidentiel', 'Sportif']

#Calcul des surfaces et du nb de batiment par usage
for usage in usages_batiments:
    join.loc[
        join["USAGE1"] == usage, f"Surf {usage}"
        ] = join.loc[join["USAGE1"] == usage].area
    join.loc[
        join["USAGE1"] != usage, f"Surf {usage}"
        ] = 0
    aggfunc[f"Surf {usage}"] = "sum"
    join.loc[
        join["USAGE1"] == usage, f"Nb {usage}"
        ] = 1
    join.loc[
        join["USAGE1"] != usage, f"Nb {usage}"
        ] = 0
    aggfunc[f"Nb {usage}"] = "sum"

#Agrégation des polygones qui appartenaient au même polygone ocsge initial
ocsge_bati = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

ocsge_bati.to_file("ocsge_bati.gpkg", layer='ocsge_bati', driver="GPKG")

# #Apparemment il manque des polygones
# polys_manquants = ocsge.overlay(ocsge_bati, how='difference')
# polys_manquants.to_file("polys_manquants.gpkg", layer='polys_manquants', driver="GPKG")

# columns_to_drop = []
# for column in polys_manquants.columns:
#     if not column in aggfunc.keys() and column!='geometry':
#         columns_to_drop.append(column)
# polys_manquants = polys_manquants.drop(columns=columns_to_drop)
# for column in aggfunc.keys():
#     if not column in polys_manquants.columns:
#         polys_manquants.loc[:, column]=0

# ocsge_bati = ocsge_bati.concat(polys_manquants)
# ocsge_bati.to_file("ocsge_bati.gpkg", layer='ocsge_bati', driver="GPKG")

for k, usage in enumerate(usages_batiments):
    f = plt.figure(figsize=(17,13))
    for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
        for j, prefix in enumerate(["Surf", "Nb"]):
            column = f"{prefix} {usage}"
            ax = plt.subplot(2, 4, 1+i+4*j)
            plt.title(f"{US} {column}")
            plt.yscale("log")
            ocsge_bati[ocsge_bati["CODE_US"]==US].loc[:, column].hist(
                ax=ax, bins=30)
            # ax.hist(ocsge_bati[ocsge_bati["CODE_US"]==US].loc[:, column],
            #         density=True)
            # plt.ylim(1e-5,1)
plt.show()

#%%Statistiques sur les bâtiments
lims = [2000, 50]
f = plt.figure(figsize=(17,13))
for j, prefix in enumerate(["Surf", "Nb"]):
    columns_to_select = [
        f"{prefix} {usage}" for usage in usages_batiments]
    total_prefix = ocsge_bati.loc[:, columns_to_select].sum(axis=1)
    for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
        ax = plt.subplot(2, 4, 1+i+4*j)
        plt.title(f"{US} total {prefix}")
        plt.yscale("log")
        total_prefix[
            np.bitwise_and(ocsge_bati["CODE_US"]==US, (total_prefix<lims[j]))
            ].hist(ax=ax, bins=100)
        # plt.xlim(0,100)

for k, usage in enumerate(usages_batiments):
    f = plt.figure(figsize=(17,13))
    for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
        for j, prefix in enumerate(["Surf", "Nb"]):
            column = f"{prefix} {usage}"
            ax = plt.subplot(2, 4, 1+i+4*j)
            plt.title(f"Densite {US} {column}")
            plt.yscale("log")
            (ocsge_bati[ocsge_bati["CODE_US"]==US].loc[:, column]/\
             ocsge_bati[ocsge_bati["CODE_US"]==US].area).hist(
                ax=ax, bins=30)
            # ax.hist(ocsge_bati[ocsge_bati["CODE_US"]==US].loc[:, column],
            #         density=True)
            # plt.ylim(1e-5,1)
plt.show()

#%%Intersection avec les communes de la bd topo
#pour récupérer la population de la commune

ocsge_bati = gpd.read_file(
    "ocsge_bati.gpkg",
    layer='ocsge_bati',
    driver="GPKG")



communes = gpd.read_file(path_communes_bd_topo)
join = ocsge_bati.overlay(communes,
                          how='intersection',
                          keep_geom_type=True)

#Le cas des polygones à cheval entre plusieurs communes
aggfunc = {"POPULATION": "mean"}
for col in ocsge_bati.columns:
    if col not in ["ID_1", "geometry"]:
        aggfunc[col] = "first"

ocsge_bati_population = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )
ocsge_bati_population.to_file(
    "ocsge_bati_population.gpkg",
    layer='ocsge_bati_population',
    driver="GPKG")


#%%Plus de géométrie

couche = gpd.read_file(
    "ocsge_bati_population.gpkg",
    layer='ocsge_bati_population',
    driver="GPKG").set_index("ID_1")




#%% Ajout de OSO

couche["OSO"] = couche.geometry.apply(recuperer_oso)

couche.to_file(
    "ocsge_bati_population.gpkg",
    layer='ocsge_bati_population',
    driver="GPKG"
    )

#%%Chargement d'OSM
osm = gpd.read_file(path_osm)

osm = osm.drop(np.where(osm.loc[:,"type"]=="boundary")[0])

classify_osm(osm,
            "OSM_to_OCSGE.csv",
             os.path.join(
                 savepath_classified_osm, "polygons"
                 ),
             "condition_code_polygons"
             )

#US1.1
list_agricol_build_val = ['silo', 'farm', 'greenhouse', 'farm_auxiliary', 'barn',
                     'cowshed', 'slurry_tank', 'stable', 'sty']

list_agricol_craft_val = ['grinding_mill', 'fruit_press'
                      'oil_mill', 'winery', 'agricultural_engines']

list_agricol_landuse_val = [
    'farmland', 'farmyard',  'meadow',
    'orchard', 'vineyard', 'greenhouse_horticulture']

#US.1.2
list_forestry_craft_val = ['sawmill']
list_forestry_landuse_val = ['forest']#Warning not always meaning it is used for forestry


#US1.3
list_extraction_landuse_val = ['quary', 'salt_pond']
list_extraction_man_made_val = [
    'adit', 'mineshaft', 'offshore_platform', 'petroleum_well']

#US1.4
list_US1_4_landuse_val = ['aquaculture']

#US3
list_tertiary_build_val = [
    'public', 'hospital', 'church', 'school', 'convent',
    'grandstand', 'commercial', 'chapel', 'civic', 'retail',
    'fire_station', 'service', 'dormitory', 'university', 'hotel',
    'government', 'office', 'cathedral', 'castle', 'sports_centre',
    'shop', 'kindergarten', 'parking', 'kiosk', 'warehouse', 'presbitary',
    'kingdom_hall', 'monastery', 'mosque', 'religious', 'shrine', 'temple',
    'synagogue', 'college', 'pavilion', 'riding_hall', 'sports_hall',
    'stadium', 'supermarket', 'wayside_shrine'
    ]#Parking is supposed to be a parking building

list_tertiary_landuse_val = [
    'religious', 'cemetery', 'commercial', 'retail',
    'education', 'recreation_ground', 'churchyard',
    'animal_keeping', 'healthcare', 'institutional',
    'winter_sports'
    ]

list_tertiary_craft_val = [
    'atelier',' chimney_sweeper', 'cleaning', 'electronics_repair',
    'electrician', 'gardener', 'interior_decorator', 'locksmith',
    'scaffolder', ' optician', 'painter', 'photographer',
    'photographic_laboratory', ' piano_tuner', 'plumber',
    'sculptor'
    ]

list_tertiary_amenity_val = [
    'bar', 'biergarten', 'cafe', 'fast_food', 'food_court',
    'ice_cream', 'pub', 'restaurant', 'college', 'driving_school',
    'kindergarten', 'language_school', 'library', 'toy_library',
    'research_institute', 'training', 'music_school', 'school',
    'traffic_park', 'university', 'atm', 'bank', 'fuel', 'bureau_de_change',
    'baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home',
    'pharmacy', 'social_facility', 'veterinary', 'arts_centre', 'brothel',
    'casino', 'cinema', 'community_centre', 'conference_centre', 'events_venue',
    'exhibition_centre', 'gambling', 'love_hotel', 'music_venue', 'nightclub',
    'planetarium', 'social_centre', 'stripclub', 'studio', 'swingerclub',
    'theatre', 'courthouse', 'fire_station', 'police', 'post_depot',
    'post_office', 'prison', 'ranger_station', 'townhall',
    'childcare', 'crematorium', 'dive_centre', 'funeral_hall',
    'grave_yard', 'internet_cafe', 'kitchen', 'monastery',
    'place_of_mourning', 'place_of_worship', 'public_bath',
    'car_rental', 'bar;cafe;pub', 'concert_hall', 'cooking_school',
    'coworking_space', 'dancing_school', 'dojo', 'dormitory',
    'driver_training', 'hookah_lounge', 'sport_school', 'workshop'
    ]



list_other_tertiary_keys = [
    "attraction", "club", "museum", "vending_machine",
    "cuisine", "golf", "boules", "emergency", "government",
    "religion", "commercial", "cemetery", "fitness_station",
    "horse_riding", "hotel", "miniature_golf", "pitch", "zoo"
    ]

#US2

list_industrial_build_val = [
    'industrial', 'storage_tank', 'digester']

list_indust_man_made_val = [
    'kiln', 'works']

list_other_industrial_keys = [
    "industrial"]

list_not_indus_craft_val = (
    list_tertiary_craft_val +
    list_agricol_craft_val +
    list_forestry_craft_val +
    [None]
    )

#US4.1.1

list_road_amenity_val = [
    'parking', 'parking_entrance',
    'parking_space', 'taxi'
    ]

#US4.1.2
list_railway_landuse_val = ['railway']
list_railway_railway_val = ['rail', 'station', 'platform', 'halt']
#We do not keep all values as some are for public transport

#US4.1.3 #aeroway => all

#US4.1.4 navigable waterways
list_US_4_1_4_landuse_val = ['port']
list_US_4_1_4_boat_val = ['yes']

#US4.3
list_public_utility_networks_building_val = [
    'transformer_tower', 'water_tower'
    ]
list_public_utility_networks_man_made_val = [
    'pumping_station', 'storage_tank', 'water_tower',
    'water_works', 'wastewater_plant'
    ]

#US5
list_residential_build_val = [
    'apartments', 'house', 'residential', 'garage', 'garages',
    'terrace', 'semidetached_house', 'hut', 'houseboat',
    'shed', 'static_caravan'
    ]

list_residential_landuse_val = [
    'allotments', 'residential'
    ]

list_other_residential_keys = [
    "community", "allotments"]

#US6.1
list_in_transition_building_val = [
    'construction'
    ]

list_in_transition_landuse_val = [
    'construction'
    ]

#US6.2
list_abandoned_landuse_val = ['brownfield']
#abandoned:landuse and not landuse
#disused:landuse

list_other_abandonned_keys = [
    "abandoned", "disused"]

#-------------------------
#Selection of OSM polygons
#-------------------------


agricol_buildings = osm[
    (osm.loc[:,"building"].isin(
        list_agricol_build_val))
    ]

agricol_landuse = osm[
    (osm.loc[:,"landuse"].isin(
        list_agricol_landuse_val))
    ]

industrial_buildings = osm[
    (osm.loc[:,"building"].isin(
        list_industrial_build_val)) |
    (~ osm.loc[:,"craft"].isin(
        list_not_indus_craft_val)) |

    (osm["man_made"].isin(
        list_indust_man_made_val)) |\
    osm["other_tags"].apply(
        lambda x: (
            x is not None and (
                    np.any([
                        '"{key}"=>' in x for key in list_other_industrial_keys])
                    )
                )
            )
    ]

industrial_landuse = osm[
    (osm.loc[:,"landuse"]=="industrial")
    ]

tertiary_buildings = osm[
    osm.loc[:,"building"].isin(
        list_tertiary_build_val
        ) |\
    osm.loc[:, "amenity"].isin(
        list_tertiary_amenity_val
        ) |\
    ~osm.loc[:, "tourism"].isnull() |\
    ~osm.loc[:, "shop"].isnull() |\
    ~osm.loc[:, "sport"].isnull() |\
    ~osm.loc[:, "office"].isnull() |\
    ~osm.loc[:, "military"].isnull() |\
    osm.loc[:, "craft"].isin(
        list_tertiary_craft_val
        ) |\
    (~osm.loc[:, "leisure"].isnull() & (
        osm["other_tags"].apply(
        lambda x: (
            x is not None and ~(
                    ('"access"=>"private"' in x) |\
                    ('"garden:type"=>"residential"'in x)
                    )
                )
            )
        )
    ) |\
    osm["other_tags"].apply(
    lambda x: (
        x is not None and (
                np.any([
                    '"{key}"=>' in x for key in list_other_tertiary_keys])
                )
            )
        )
    ]

tertiary_landuse = osm[
    osm.loc[:, "landuse"].isin(
        list_tertiary_landuse_val
        )
    ]

residential_buildings = osm[
    osm.loc[:,"building"].isin(
        list_residential_build_val
        ) |
    (~osm.loc[:, "leisure"].isnull() & (
        osm["other_tags"].apply(
        lambda x: (
            x is not None and (
                    ('"access"=>"private"' in x) |\
                    ('"garden:type"=>"residential"'in x)
                    )
                )
            )
        )
    ) |\
    osm["other_tags"].apply(
    lambda x: (
        x is not None and (
                np.any([
                    '"{key}"=>' in x for key in list_other_residential_keys])
                ) |
                ('"garden:type"=>"community"' in x)
            )
        )
    ]

residential_landuse = osm[
    osm.loc[:, "landuse"].isin(
        list_residential_landuse_val
        )
    ]

list_building_not_to_use = [
    None, 'construction', 'transformer_tower',
    'water_tower', 'train_station'
    ] + list_industrial_build_val + list_agricol_build_val + list_tertiary_build_val +\
    list_residential_build_val


indifferenciated_buildings = osm[
    (~osm.loc[:,"building"].isin(
        list_building_not_to_use
        ) | ~osm["historic"].isnull()) &
    osm.loc[:, "amenity"].isnull() &\
    osm.loc[:,"craft"].isnull() &\
    osm.loc[:, "leisure"].isnull() &\
    osm.loc[:, "tourism"].isnull() &\
    osm.loc[:, "shop"].isnull() &\
    osm.loc[:, "sport"].isnull() &\
    osm.loc[:, "office"].isnull() &\
    osm.loc[:, "military"].isnull()
    ]

#%%Intersection avec OSM


ocsge_bati_population = gpd.read_file(
    "ocsge_bati_population.gpkg",
    layer='ocsge_bati_population',
    driver="GPKG")

ocsge_bati_population["HAUTEUR"] = ocsge_bati_population["HAUTEUR"].fillna(0)

couches = [industrial_buildings,
           residential_buildings,
           agricol_buildings,
           tertiary_buildings,
           indifferenciated_buildings]

names = ["industrial",
         "resid",
         "agricol",
         "tertiary",
         "indif"
         ]


join = ocsge_bati_population.copy()#.to_crs(agricol_buildings.crs)
# join = join.set_index("ID_1")
join = join.drop(columns=list(set(join.columns)-set(["ID_1", "geometry"])))

for i, couche in enumerate(couches):
    print(i)
    # print(join.crs)
    # print(np.all(join.is_valid))

    join_copy = join.copy()
    #On vire les colonnes qui ne nous intéressent pas
    for col in couche.columns:
        if col not in ["osm_id", "geometry"]:
            couche = couche.drop(columns=col)

    #Il y a des problèmes de géométrie dans certaines couches



    # couche.geometry = couche.buffer(0)
    # join.geometry = [make_valid(ob) for ob in join.geometry]

    couche  = couche.to_crs(ocsge_bati_population.crs)


    aggfunc = {}
    for column in join.columns:
        if column not in ["geometry"]:
            aggfunc[column] = "first"
    aggfunc[f"osm nb {names[i]}"] = "sum"
    aggfunc[f"osm surf {names[i]}"] = "sum"


    intersect_ok = False
    print(f"intersection de {names[i]}")
    try:
        join = couche.overlay(join,
                              how='intersection',
                              keep_geom_type=True)

    except errors.TopologicalError:
        print(f"{names[i]} n'a pas pu être intersecté, tentative de simplifier les coordonnées")
        try:
            couche.geometry = [
                loads(dumps(geom, rounding_precision=3)) for geom in couche.geometry]
            join.geometry = [
                loads(dumps(geom, rounding_precision=3)) for geom in join.geometry]
            join = couche.overlay(join,
                                  how='intersection',
                                  keep_geom_type=True)

        except errors.TopologicalError:
            print(f"{names[i]} n'a pas pu être intersecté de nouveau. On n'y prend donc pas en compte.")

        else:
            print("C'est bon cette fois !")
            intersect_ok = True

    else:
        intersect_ok = True

    if intersect_ok:
        print(f"agrégation de {names[i]}")
        # join = join.to_crs(ocsge_bati_population.crs)

        intersected = ~join.loc[:,"osm_id"].isna()

        join.loc[intersected, f"osm nb {names[i]}"] = 1
        join.loc[~intersected, f"osm nb {names[i]}"] = 0

        join.loc[intersected, f"osm surf {names[i]}"] = join.loc[intersected].area
        join.loc[~intersected, f"osm surf {names[i]}"] = 0

        join = join.drop(columns="osm_id")

        #Enlever les batiments hors de l'emprise de l'OCSGE
        # join = join.drop(index=(np.where(join.loc[:, "ID_1"].isna())[0]))

        #Agrégation des polygones qui appartenaient au même polygone ocsge initial
        join = join.dissolve(
            by="ID_1",
            aggfunc=aggfunc
            )

        not_intersected = list(set(join_copy["ID_1"]) - set(join["ID_1"]))
        join = join.append(join_copy[join_copy["ID_1"].isin(not_intersected)])
        join[[
            f"osm nb {names[i]}",
            f"osm surf {names[i]}"]
            ] = join[[
                f"osm nb {names[i]}",
                f"osm surf {names[i]}"]
                ].fillna(0)
    #Sinon join est toujours l'ancien join et la couche n'est pas prise en compte



print("intersection faite")

join = join.to_crs(ocsge_bati_population.crs)


join = join.set_index("ID_1")

join[ocsge_bati_population.set_index(
    "ID_1").columns] = ocsge_bati_population.set_index("ID_1")

join.to_file(
    "ocsge_bati_population_osm.gpkg",
    layer='ocsge_bati_population_osm',
    driver="GPKG")

# aggfunc = {}
# for column in ocsge_bati_population.columns:
#     if column not in ["geometry"]:
#         aggfunc[column] = "first"

# for name in names:
#     aggfunc[f"osm nb {name}"] = "sum"
#     aggfunc[f"osm surf {name}"] = "sum"
#%% OSM polygons new version
list_LU_polygons = [
    "LU1_1", "LU1_2", "LU2", "LU3",
    "LU4_1_1", "LU4_1_2", "LU4_1_3", "LU4_3",
    "LU5", "LU6_1", "LU6_2",
    "LU1_1_landuse", "LU2_landuse", "LU3_landuse", "LU5_landuse"]
for LU in list_LU_polygons:
    print(LU)
    osm_polygons  = gpd.read_file(
        os.path.join(
            savepath_classified_osm, "polygons", f"{LU}.gpkg"
            )
        ).to_crs(crs=ocsge.crs)
    new_cols = {
        f"osm_{LU}_surface" : lambda x:x.area
        }
    aggfunc = {
        f"osm_{LU}_surface" : "sum"
        }
    if "landuse" not in LU:
        new_cols[f"osm_{LU}_number"] = lambda x:1
        aggfunc[f"osm_{LU}_number"] = "sum"

    if (~ocsge.is_valid).any():
        ocsge.geometry = [make_valid(ob) for ob in ocsge.geometry]

    ocsge = add_vector_layer(
        ocsge,
        osm_polygons,
        aggfunc,
        None,
        LU,
        new_cols=new_cols
        )

ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=[
                                   x
                                   for LU in list_LU_polygons
                                   for x in [
                                           f"osm_{LU}_number",
                                           f"osm_{LU}_surface"]
                                   if "landuse_number" not in x
                                   ],
                               quali_cols=[],
                               )
# ocsge.to_file("plus_osm_poly_updated.gpkg")
ocsge.to_file("E://osm_polygons.gpkg")
#%% OSM ways new version
osm_lines = gpd.read_file(
    path_osm_lines
    )

classify_osm(osm_lines,
            "OSM_to_OCSGE.csv",
             os.path.join(
                 savepath_classified_osm, "lines"
                 ),
             "condition_code_points"
             )

list_LU_lines = [
    "LU4_1_1", "LU4_1_2", "LU4_1_3"
    ]

for LU in list_LU_lines:
    print(LU)
    osm_lines  = gpd.read_file(
        os.path.join(
            "D:", "OSM_69", "lines", f"{LU}.gpkg"
            )
        ).to_crs(crs=ocsge.crs)
    new_cols = {
        f"osm_{LU}_length" : lambda x:x.length
        }
    aggfunc = {
        f"osm_{LU}_length" : "sum"
        }

    ocsge = add_vector_layer(
        ocsge,
        osm_lines,
        aggfunc,
        None,
        LU,
        new_cols=new_cols
        )

ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=[
                                   f"osm_{LU}_length"
                                   for LU in list_LU_lines
                                   ],
                               quali_cols=[],
                               )
# ocsge.to_file("E://osm_lines.gpkg")
ocsge.to_file("plus_osm_lines_updated.gpkg")
#%% OSM points new version
list_LU_pts = ["LU1_1", "LU2", "LU3", "LU4_1_1", "LU4_1_2", "LU4_1_3", "LU5"]
for LU in list_LU_pts:
    osm_pts  = gpd.read_file(
        os.path.join(
            savepath_classified_osm, "points", f"{LU}.gpkg"
            )
        ).to_crs(crs=ocsge.crs)
    if (~ocsge.is_valid).any():
        ocsge.geometry = [make_valid(ob) for ob in ocsge.geometry]
    ocsge = add_points(ocsge, osm_pts, f"pts_{LU}")
ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=[
                                   f"pts_{LU}"
                                   for LU in list_LU_pts
                                   ],
                               quali_cols=[],
                               )
ocsge.to_file("plus_osm_pts_updated.gpkg")
#ocsge.to_file("E://osm_pts.gpkg")

#%% Ajout des points OSM

couche = gpd.read_file(
    "ocsge_bati_population_osm.gpkg",
    layer='ocsge_bati_population_osm',
    driver="GPKG")

osm_pts = gpd.read_file(
    path_osm_pts
    ).to_crs(crs=couche.crs)


us_4_1_1_pts = osm_pts[
    (~osm_pts.loc[:,"highway"].isna())|\
    (osm_pts["man_made"].isin(['gantry'])) |\
    (osm_pts["other_tags"].apply(
        lambda x: (
            x is not None and (
                np.any([f'"amenity"=>"{value}"' in x \
                        for value in [
                            "parking",
                            "parking_entrance",
                            "parking_space",
                            "taxi"
                            ]
                        ]
                    )
                )
            )
        )
    )
    ]

osm_pts = osm_pts.drop(
    np.where(
        ~osm_pts.loc[:,"barrier"].isna()|\
        ~osm_pts.loc[:,"highway"].isna()|\
        ~osm_pts.loc[:,"place"].isna()
        )[0])

industrial_pts = osm_pts[
    (osm_pts["man_made"].isin(['kiln', 'works'])) |\
    (osm_pts["other_tags"].apply(
        lambda x: (
            x is not None and (
                ("pipeline" in x) |\
                ("industrial" in x) |\
                np.any([f'"building"=>"{value}"' in x \
                        for value in list_industrial_build_val]) |\
                (("craft" in x) &\
                     ~np.any([f'"craft"=>"{value}"' in x \
                        for value in list_not_indus_craft_val])) |\
                np.any([
                    f'"{key}=>"' in x for key in list_other_industrial_keys])
                )
            )
        )
    )]

tertiary_pts = osm_pts[
    (osm_pts["other_tags"].apply(
        lambda x: (
            x is not None and (
                ("shop" in x) |\
                (("leisure" in x) & ~(
                    ('"access"=>"private"' in x) |\
                    ('"garden:type"=>"residential"'in x) |\
                    ('"leisure"=>"common"' in x))
                ) |\
                ("sport" in x) |\
                ("tourism" in x) |\
                ("military" in x) |\
                ('"tower:type"=>"communication"' in x) |\
                ("office" in x) |\
                np.any([f'"building"=>"{value}"' in x \
                        for value in list_tertiary_build_val]) |\
                np.any([f'"landuse"=>"{value}"' in x \
                        for value in list_tertiary_landuse_val]) |\
                np.any([f'"amenity"=>"{value}"' in x \
                        for value in list_tertiary_amenity_val]) |\
                (("craft" in x) &\
                     np.any([f'"craft"=>"{value}"' in x \
                        for value in list_tertiary_craft_val])) |\
                np.any([
                    f'"{key}=>"' in x for key in list_other_tertiary_keys])
                )==1
            )
        )
    )]

residential_pts = osm_pts[
    (osm_pts["other_tags"].apply(
        lambda x: (
            x is not None and (
                (("leisure" in x) & (
                    ('"access"=>"private"' in x) |\
                    ('"garden:type"=>"residential"'in x)) &\
                    ~('"leisure"=>"common"' in x)
                ) |\
                np.any([f'"building"=>"{value}"' in x \
                        for value in list_residential_build_val]) |\
                np.any([f'"landuse"=>"{value}"' in x \
                        for value in list_residential_landuse_val]) |\
                np.any([
                    f'"{key}=>"' in x for key in list_other_residential_keys])
                )==1
            )
        )
    )]

cols_name = ["osm_pts_us2", "osm_pts_us3", "osm_pts_us5"]
for i, couche_pts in enumerate([industrial_pts, tertiary_pts, residential_pts]):
    couche = add_points(couche, couche_pts, cols_name[i])

couche.to_file(
    "ocsge_bati_population_osm.gpkg",
    layer='ocsge_bati_population_osm',
    driver="GPKG")

#%%Statistique des batiments OSM

ocsge_bati_population_osm = gpd.read_file(
    "ocsge_bati_population_osm.gpkg",
    layer='ocsge_bati_population_osm',
    driver="GPKG")

for k, usage in enumerate(["resid",
                           "agricol",
                           "industrial",
                           "tertiary",
                           "indif"]):

    f = plt.figure(figsize=(17,13))
    for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
        for j, prefix in enumerate(["osm surf", "osm nb"]):
            column = f"{prefix} {usage}"
            ax = plt.subplot(2, 4, 1+i+4*j)
            plt.title(f"{US} {column}")
            plt.yscale("log")
            ocsge_bati_population_osm[
                ocsge_bati_population_osm["CODE_US"]==US].loc[:, column].hist(
                    ax=ax, bins=30)
            # ax.hist(ocsge_bati[ocsge_bati["CODE_US"]==US].loc[:, column],
            #         density=True)
            # plt.ylim(1e-5,1)
plt.show()

#%% ZA BD topo

ocsge_bati_population_osm = gpd.read_file(
    "ocsge_bati_population_osm.gpkg",
    layer='ocsge_bati_population_osm',
    driver="GPKG")

za = gpd.read_file(path_za_bd_topo)

#Mappage des catégories vers celles qui nous intéressent

za.loc[:, "za_us1_1"] = 1*(za.loc[:, "NATURE"].isin([
    "Divers agricole", "Elevage"
    ]))

za.loc[:, "za_us1_3"] = 1*(za.loc[:, "NATURE"].isin([
    "Carrière", "Marais salant", "Mine"
    ]))

za.loc[:, "za_us1_4"] = 1*(za.loc[:, "NATURE"].isin([
    "Aquaculture"
    ]))

za.loc[:, "za_us2"] = 1*(za.loc[:, "NATURE"].isin([
    "Usine de production d'eau potable", "Centrale électrique",
    "Divers industriel", "Usine", "Zone industrielle"
    ]))

za.loc[:, "za_us3"] = 1*(za.loc[:, "CATEGORIE"].isin([
    "Sport", "Administratif ou militaire",
    "Science et enseignement",
    "Culture et loisirs", "Religieux", "Santé"]) |\
    za.loc[:, "NATURE"].isin([
        "Marché", "Divers commercial", "Déchèterie", "Haras"
        ]))

za.loc[:, "za_us4_3"] = 1*(za.loc[:, "NATURE"].isin([
    "Station de pompage", "Station d'épuration"
    ]))


join = ocsge_bati_population_osm.overlay(
    za,
    how='union',
    keep_geom_type=False)
liste_za = ["za_us1_1", "za_us1_3", "za_us1_4", "za_us2", "za_us3", "za_us4_3"]
join[liste_za] = join[liste_za].fillna(0)

f = lambda x: np.average(x, weights=join.loc[x.index].area)

#Le cas des polygones à cheval entre plusieurs za
aggfunc = {za_us_i: f
           for za_us_i in liste_za}
# for col in ocsge_bati_population_osm.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

aggr = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

ocsge_bati_population_osm_za = aggr.copy()

ocsge_bati_population_osm_za[
    ocsge_bati_population_osm.set_index("ID_1").columns
    ] = ocsge_bati_population_osm.set_index("ID_1")

ocsge_bati_population_osm_za.to_file(
    "ocsge_bati_population_osm_za.gpkg",
    layer='ocsge_bati_population_osm_za',
    driver="GPKG")

#%% ERP BD topo

ocsge_bati_population_osm_za = gpd.read_file(
    "ocsge_bati_population_osm_za.gpkg",
    layer='ocsge_bati_population_osm_za',
    driver="GPKG")



ERP = gpd.read_file(path_ERP_bd_topo)
ERP = ERP.drop(columns=ERP.columns[1:-1])
ERP = ERP.rename(columns={"ID" : "ERP"})


join = ocsge_bati_population_osm_za.sjoin(
    ERP,
    how='inner',
    predicate="intersects"
    )

aggfunc = {"ERP": "count"}


ocsge_bati_population_osm_za_ERP = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

ocsge_bati_population_osm_za_ERP[
    ocsge_bati_population_osm_za.columns[1:]
    ] = ocsge_bati_population_osm_za.set_index("ID_1")

#Il manque les polygones non intersectés
idx_non_intersect = list(set(ocsge_bati_population_osm_za.set_index("ID_1").index) - set(ocsge_bati_population_osm_za_ERP.index))
non_intersect = ocsge_bati_population_osm_za.set_index("ID_1").loc[idx_non_intersect]
non_intersect["ERP"] = 0
ocsge_bati_population_osm_za_ERP = ocsge_bati_population_osm_za_ERP.append(non_intersect)

ocsge_bati_population_osm_za_ERP.to_file(
    "ocsge_bati_population_osm_za_ERP.gpkg",
    layer='ocsge_bati_population_osm_za_ERP',
    driver="GPKG")

#%%

ocsge_bati_population_osm_za_ERP = gpd.read_file(
    "ocsge_bati_population_osm_za_ERP.gpkg",
    layer='ocsge_bati_population_osm_za_ERP',
    driver="GPKG")


f = plt.figure(figsize=(17,13))
for i, US in enumerate(['US2', 'US3', 'US5', 'US235']):
    for j, column in enumerate(['ERP', 'za_us2',
                                'za_us3']):
        ax = plt.subplot(3, 4, 1+i+4*j)
        plt.title(f"{US} {column}")
        plt.yscale("log")
        plt.ylim(bottom=0.8, top=1e5)
        ocsge_bati_population_osm_za_ERP[
            ocsge_bati_population_osm_za_ERP["CODE_US"]==US].loc[
                :, column].hist(ax=ax)


#%% Intersection avec CLC

couche = gpd.read_file(
    "ocsge_bati_population_osm_za_ERP.gpkg",
    layer='ocsge_bati_population_osm_za_ERP',
    driver="GPKG")

CLC = gpd.read_file(
    path_CLC
    )

join = couche.overlay(CLC,
                      how='intersection',
                      keep_geom_type=True)

join = join.set_index("ID_1")
couche = couche.set_index("ID_1")

def weighted_majority(x):
    unique = np.unique(x)
    unique_table = np.array([x==u for u in unique])
    weights=couche.loc[x.index].area
    somme = unique_table.dot(weights)
    return unique[np.argmax(somme)]

#Le cas des polygones à cheval entre plusieurs classes CLC
aggfunc = {col_CLC: weighted_majority}
# for col in couche.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

couche_plus_CLC = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

couche_plus_CLC[ couche.columns] = couche

couche_plus_CLC.to_file(
    "ocsge_bati_population_osm_za_ERP_CLC.gpkg",
    layer='ocsge_bati_population_osm_za_ERP_CLC',
    driver="GPKG")



#%%Moyenne sur un certain rayon

def mean_values_on_a_radius(couche, distance, index_name="ID_1"):
    couche_etendue = couche.copy()
    couche_etendue.geometry = couche.buffer(distance)
    join = gpd.sjoin(couche, couche_etendue, how="inner",
                     predicate="intersects", rsuffix=f"mean_{distance}m")
    join = join.rename(columns=(lambda name:name.replace("_left","")))
    aggfunc={}
    dtypes = join.dtypes


    for i, col in enumerate(join.columns):
        if "mean" in col :
            if np.issubdtype(dtypes[i], np.integer) or\
                np.issubdtype(dtypes[i], np.floating):
                aggfunc[col] = weighted_mean(join)
            else:
                aggfunc[col] = weighted_majority(join)#Valeur majoritaire
        # elif col!="geometry":
            # aggfunc[col]="first"
    print(aggfunc)

    aggr = join.dissolve(
        by=index_name,
        aggfunc=aggfunc
        )

    aggr[ couche.columns[1:]] = couche.set_index(index_name)

    return aggr

def mean_values_on_a_radius_2(couche, distance, index_name="ID_1"):
    couche_etendue = couche.copy()
    couche_etendue.geometry = couche.buffer(distance)
    join = couche.overlay(couche_etendue,
                          how='intersection',
                          keep_geom_type=False)
    # join = gpd.sjoin(couche, couche_etendue, how="inner",
    #                  predicate="intersects", rsuffix=f"mean_{distance}m")
    join = join.drop(np.where(join[index_name+"_1"]==join[index_name+"_2"])[0])
    join = join.rename(
        columns=(
            lambda name: name[:-2] if name.endswith("_1") else
                name.replace("_2",f"_mean_{distance}m")))
    aggfunc={}
    dtypes = join.dtypes



    for i, col in enumerate(join.columns):
        if "mean" in col :
            if np.issubdtype(dtypes[i], np.integer) or\
                np.issubdtype(dtypes[i], np.floating):
                aggfunc[col] = weighted_mean(join)
            else:
                aggfunc[col] = weighted_majority(join)#Valeur majoritaire
        # elif col!="geometry":
            # aggfunc[col]="first"
    print(aggfunc)

    aggr = join.groupby(
        by=index_name).agg(
            aggfunc
        )

    # aggr[ couche.columns[1:]] = couche.set_index(index_name)
    couche = couche.set_index(index_name)
    couche[ aggr.columns] = aggr

    return couche


def mean_neighbours_values(couche, quanti_cols, quali_cols):
    adj, uniques, inv = open_ocsge_adj_matrix(with_diag=False,
                                              path=path_adj_matrix)
    couche = couche.set_index("ID_1")
    df_uniques = pd.DataFrame(inv, index=uniques)

    #L'ordre dans la couche n'est pas forcément
    #le même que dans la matrice d'adjacence
    couche_meme_ordre, __ = couche.align(
        df_uniques, join='right', axis=0)

    #Colones quantitatives : valeurs moyennes
    values = np.array(couche_meme_ordre[quanti_cols])
    values_mean = adj @ values
    new_names = [name+'_mean_1m' for name in quanti_cols]
    couche_meme_ordre[new_names] = values_mean

    #Colones qualitatives : valeurs majoritaires
    if len(quali_cols)>0:
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_cols = OH_encoder.fit_transform(
            couche_meme_ordre[quali_cols].to_numpy())
        encoded_cols_mean = adj @ encoded_cols
        start = 0
        for i, col in enumerate(quali_cols):
            uniques_val = OH_encoder.categories_[i]
            nb_values = len(uniques_val)
            majo = np.argmax(encoded_cols_mean[:, start:start+nb_values], axis=1)
            couche_meme_ordre[col+'_mean_1m'] = uniques_val[majo]
            start += nb_values

    return couche_meme_ordre



ocsge_bati_population_osm_za_ERP = gpd.read_file(
    "ocsge_bati_population_osm_za_ERP.gpkg",
    layer='ocsge_bati_population_osm_za_ERP',
    driver="GPKG")


moyenne_voisin_1m = mean_values_on_a_radius_2(
    ocsge_bati_population_osm_za_ERP, 1)

moyenne_voisin_1m.to_file(
    "moyenne_voisin_1m.gpkg",
    layer='moyenne_voisin_1m',
    driver="GPKG")

#%%Statistiques sur les voisins

moyenne_voisin_1m = gpd.read_file(
    "moyenne_voisin_1m.gpkg",
    layer='moyenne_voisin_1m',
    driver="GPKG").set_index("ID_1")

f = plt.figure(figsize=(17,5))
all_unics = np.unique(moyenne_voisin_1m["CODE_US_mean_1m"])

df = pd.DataFrame(
    np.zeros((len(all_unics),4)),
    index=all_unics,
    columns=["US2", "US3", "US5", "US235"]
    )

plt.suptitle("Usage majoritaire parmi les voisins")
for i, US in enumerate(["US2", "US3", "US5", "US235"]):
    ax = plt.subplot(1, 4, 1+i)
    poly_cet_us = moyenne_voisin_1m[moyenne_voisin_1m["CODE_US"]==US]
    unic, cnts = np.unique(
        poly_cet_us["CODE_US_mean_1m"],
        return_counts=True)
    df.loc[unic, US]=cnts
    ax.pie(df.loc[:, US], labels=all_unics)
    plt.title(US)
    df.loc[unic, US]=cnts

#La même chose avec les couvertures
f = plt.figure(figsize=(17,5))
all_unics = np.unique(moyenne_voisin_1m["CODE_CS_mean_1m"])

df = pd.DataFrame(
    np.zeros((len(all_unics),4)),
    index=all_unics,
    columns=["US2", "US3", "US5", "US235"]
    )

plt.suptitle("Couverture majoritaire parmi les voisins")
for i, US in enumerate(["US2", "US3", "US5", "US235"]):
    ax = plt.subplot(1, 4, 1+i)
    poly_cet_us = moyenne_voisin_1m[moyenne_voisin_1m["CODE_US"]==US]
    unic, cnts = np.unique(
        poly_cet_us["CODE_CS_mean_1m"],
        return_counts=True)
    df.loc[unic, US]=cnts
    ax.pie(df.loc[:, US], labels=all_unics,
           colors = [
               "red", "blue", "green",
               "magenta", "cyan", "yellow",
               "brown", "grey", "orange",
               "pink", "darkgreen", "purple"
               ])
    plt.title(US)
    df.loc[unic, US]=cnts

#Leurs propres couvertures
f = plt.figure(figsize=(17,5))
all_unics = np.unique(moyenne_voisin_1m["CODE_CS"])

df = pd.DataFrame(
    np.zeros((len(all_unics),4)),
    index=all_unics,
    columns=["US2", "US3", "US5", "US235"]
    )

plt.suptitle("Couverture par usage")
for i, US in enumerate(["US2", "US3", "US5", "US235"]):
    ax = plt.subplot(1, 4, 1+i)
    poly_cet_us = moyenne_voisin_1m[moyenne_voisin_1m["CODE_US"]==US]
    unic, cnts = np.unique(
        poly_cet_us["CODE_CS"],
        return_counts=True)
    df.loc[unic, US]=cnts
    ax.pie(df.loc[:, US], labels=all_unics,
           colors = [
               "red", "blue", "green",
               "magenta", "cyan", "yellow",
               "brown", "grey", "orange",
               "pink", "darkgreen", "purple"
               ])
    plt.title(US)
    df.loc[unic, US]=cnts


#%%IRIS

couche = gpd.read_file(
    "moyenne_voisin_1m.gpkg",
    layer='moyenne_voisin_1m',
    driver="GPKG")

IRIS = gpd.read_file(
    path_iris
    )

join = couche.overlay(IRIS,
                      how='intersection',
                      keep_geom_type=True)

# join = join.set_index("ID_1")
# couche = couche.set_index("ID_1")

f = lambda x: np.average(x, weights=join.loc[x.index].area)

def weighted_majority(x):
    unique = np.unique(x)
    unique_table = np.array([x==u for u in unique])
    weights=join.loc[x.index].area
    somme = unique_table.dot(weights)
    return unique[np.argmax(somme)]

#Le cas des polygones à cheval entre plusieurs classes CLC
aggfunc = {"densite": f,
           "TYP_IRIS": weighted_majority}
# for col in couche.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

couche_plus_IRIS = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

couche = couche.set_index("ID_1")
couche_plus_IRIS[ couche.columns] = couche

couche_plus_IRIS.to_file(
    "je_vote_IRIS.gpkg",
    layer='je_vote_IRIS',
    driver="GPKG")

#%%DGFIP

couche = gpd.read_file(
    "je_vote_IRIS.gpkg",
    layer='je_vote_IRIS',
    driver="GPKG")

Foncier = gpd.read_file(
    path_foncier
    )

list_US= [
    'US11', 'US12', 'US13', 'US14', 'US15', 'US2', 'US3',
    'US411', 'US412', 'US413', 'US414', 'US415', 'US42', 'US43', 'US5',
    'US61', 'US62', 'US63', 'US66'
        ]

columns_renamer = {
    US : f"land_files_{US.replace('US', 'LU')}_area"
    for US in list_US
    }
columns_renamer["usage"] = "land_files_main_LU"


quanti_cols = [
    f"land_files_{US.replace('US', 'LU')}_area"
    for US in list_US
    ]
quanti_cols += ["nb_parcell", "nb_local"]
quali_cols = [ "land_files_main_LU"]
new_cols = quanti_cols + quali_cols

Foncier = Foncier.rename(columns=columns_renamer)

Foncier = Foncier[new_cols+["geometry"]]

Foncier = Foncier.fillna(0)

join = couche.overlay(Foncier,
                      how='intersection',
                      keep_geom_type=True)

# join = join.set_index("ID_1")
# couche = couche.set_index("ID_1")

f = lambda x: np.average(x, weights=join.loc[x.index].area)

def weighted_majority(x):
    unique = np.unique(x)
    unique_table = np.array([x==u for u in unique])
    weights=join.loc[x.index].area
    somme = unique_table.dot(weights)
    return unique[np.argmax(somme)]

#Le cas des polygones à cheval entre plusieurs classes CLC
aggfunc = {"land_files_main_LU" : weighted_majority}
for col in quanti_cols:
    aggfunc[col] = f

# aggfunc = {"nb_parcell": f,
#            "nb_local": f,
#            "usage": weighted_majority,
#            "US2": f,
#            "US3": f,
#            "US5": f}

# for col in couche.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

aggr = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

couche = couche.set_index("ID_1")

couche[new_cols] = aggr[new_cols]

couche["ID_1"] = couche.index

couche_plus_Foncier2 = mean_neighbours_values(
    couche,
    quanti_cols,
    quali_cols
    )

couche_plus_Foncier2 = couche_plus_Foncier2.fillna(0)

couche_plus_Foncier2.to_file(
    "plus_Foncier.gpkg",
    layer='plus_Foncier',
    driver="GPKG")

#%%RPG

couche = gpd.read_file(
    "plus_Foncier.gpkg",
    layer='plus_Foncier',
    driver="GPKG")


RPG = gpd.read_file(
    path_RPG
    )

RPG["surf_RPG"] = RPG.area

quanti_cols = ["surf_RPG"]
quali_cols = []
new_cols = quanti_cols + quali_cols


join = couche.overlay(RPG,
                      how='intersection',
                      keep_geom_type=True)

# join = join.set_index("ID_1")
# couche = couche.set_index("ID_1")

f = lambda x: np.average(x, weights=join.loc[x.index].area)

def weighted_majority(x):
    unique = np.unique(x)
    unique_table = np.array([x==u for u in unique])
    weights=join.loc[x.index].area
    somme = unique_table.dot(weights)
    return unique[np.argmax(somme)]

#Le cas des polygones à cheval entre plusieurs classes
aggfunc = {"surf_RPG": f}

# for col in couche.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

aggr = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

couche = couche.set_index("ID_1")

couche[new_cols] = aggr[new_cols]

couche["ID_1"] = couche.index

couche_plus_RPG = mean_neighbours_values(
    couche,
    quanti_cols,
    quali_cols
    )

couche_plus_RPG = couche_plus_RPG.fillna(0)

couche_plus_RPG.to_file(
    "plus_RPG.gpkg",
    layer='plus_RPG',
    driver="GPKG")

#%%Images Orthos RVB et PIR

pas = 10000
i0 = 0

# couche = gpd.read_file(
#     "plus_RPG.gpkg",
#     layer='plus_RPG',
#     driver="GPKG")

# couche = gpd.read_file(
#     f"plus_image_{i0-5000}.gpkg",
#     layer='plus_image',
#     driver="GPKG")

couche = gpd.read_file(
    "E:\\moyenne_voisin_1m.gpkg")

opened_dalles = open_shapefile_dalle()


# for i in range(i0, len(couche), pas):
#     print(f"{i}/{len(couche)}", datetime.datetime.now())
#     couche.loc[i:i+pas-1, [
#         "meanR", "meanV", "meanB",
#         "stdR", "stdV", "stdB"]
#         ] = couche.loc[i:i+pas-1].apply(recuperer_Ortho, axis=1)
#     print("RVB done", datetime.datetime.now())
#     couche.loc[i:i+pas-1,
#                ["meanPIR", "stdPIR"]
#                ] = couche.loc[i:i+pas-1].apply(recuperer_IRC, axis=1)
#     print("PIR done", datetime.datetime.now())
#     couche.to_file(
#     f"plus_image_{i}.gpkg",
#     layer='plus_image',
#     driver="GPKG"
#     )

for i in range(i0, len(couche), pas):
    print(f"{i}/{len(couche)}", datetime.datetime.now())
    couche.loc[i:i+pas-1, [
        "meanPIR", "meanR", "meanV", "meanB",
        "stdPIR", "stdR", "stdV", "stdB"]
        ] = couche.loc[i:i+pas-1].apply(
            recuperer_RVBI,
            axis=1,
            args=[opened_dalles]
            )
    couche.to_file(
    f"E:\\plus_image_{i}.gpkg",
    layer='plus_image',
    driver="GPKG"
    )

print("Finito !")
#%%Voisinage pour les images
radiometric_cols = ['meanR', 'meanV', 'meanB', 'meanPIR',
                    'stdR', 'stdV', 'stdB', 'stdPIR']

ocsge = gpd.read_file(
    "plus_image.gpkg",
    layer='plus_image',
    driver="GPKG")

ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=radiometric_cols,
                               quali_cols=[],
                               )

ocsge.to_file(
    "plus_image.gpkg",
    layer='plus_image',
    driver="GPKG"
    )


#%%Carroyage INSEE


couche = gpd.read_file(
    "plus_image.gpkg",
    layer='plus_image',
    driver="GPKG")

carroyage_INSEE = gpd.read_file(
    path_carroyage_INSEE
    )

carroyage_INSEE["Mean_snv"] = carroyage_INSEE["Ind_snv"] / carroyage_INSEE["Ind"]

join = couche.overlay(carroyage_INSEE,
                      how='union',
                      keep_geom_type=True)

# join = join.set_index("ID_1")
couche = couche.set_index("ID_1")

#Le cas des polygones à cheval entre plusieurs carreaux
aggfunc = {"Ind": weighted_mean(join),#Population of the 200m x 200m square
           "Mean_snv": weighted_mean(join), #Mean level of life
           "Log_soc": weighted_mean(join) #Number of social housing units
           }
# for col in couche.columns:
#     if col not in ["ID_1", "geometry"]:
#         aggfunc[col] = "first"

couche_plus_INSEE = join.dissolve(
    by="ID_1",
    aggfunc=aggfunc
    )

couche_plus_INSEE[ couche.columns] = couche

couche_plus_INSEE = couche_plus_INSEE.reset_index()
couche_plus_INSEE = mean_neighbours_values(couche_plus_INSEE,
                               quanti_cols=["Ind",
                                            "Mean_snv",
                                            "Log_soc"],
                               quali_cols=[],
                               )

couche_plus_INSEE.to_file(
    "plus_INSEE.gpkg",
    layer='plus_INSEE',
    driver="GPKG")

#%%Autres couches BD TOPO
ocsge = gpd.read_file(
    "E:\\plus_INSEE.gpkg",
    layer='plus_INSEE',
    driver="GPKG")


#Surfacic layers
for (nom, path_vector_layer, filters, aggfunc) in [
        (
            "aerodromes",
            path_aerodrome,
            lambda x: x["is_aerodromes"],
            {"frac_surf_aerodromes": weighted_mean}
            ),
        (
            "cimetieres",
            path_cimetieres_bd_topo,
            lambda x: x["is_cimetieres"],
            {"frac_surf_cimetieres": weighted_mean}
            ),
        (
            "hydro",
            path_river,
            lambda x:
                (x.area > 200)#Filter on area
                &
                (x["POS_SOL"].astype("uint64")>=0)#Remove underground rivers
                &
                (~x["NATURE_hydro"].isin(["Ecoulement hyporhéique",
                                   "Glacier, névé",
                                   "Ecoulement phréatique",
                                   "Ecoulement karstique"]))
                &
                (x["PERSISTANC"]=="Permanent"),
            {
                "frac_surf_hydro": weighted_mean,
                "NATURE_hydro":weighted_majority
                }
        )
        ]:
    vector_layer = gpd.read_file(
        path_vector_layer
    )
    vector_layer[f"is_{nom}"]=True

    if nom == "hydro":
        vector_layer = vector_layer.rename(
            columns={"NATURE" : "NATURE_hydro"}
            )

    vector_layer = vector_layer[filters(vector_layer)]#filtering

    ocsge = add_vector_layer(
        ocsge,
        vector_layer,
        aggfunc,
        f"plus_{nom}.gpkg",
        f"plus_{nom}",
        "union",
        {f"frac_surf_{nom}": lambda join: frac_surf_if_condition(join, f"is_{nom}")}
        )

ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=[f"frac_surf_{nom}"
                                            for nom in [
                                                    "aerodromes",
                                                    "cimetieres",
                                                    "hydro"
                                                    ]
                                            ],
                               quali_cols=["NATURE_hydro"],
                               )
ocsge = ocsge.reset_index()

#Linear layers
#filters to avoid in project roads,...
for (nom, path_vector_layer, filters) in [
        ("routes",
          path_routes,
          lambda x:
              (x["ETAT"]!="En projet")
              &
              (x["NATURE"].isin([
                  "Bretelle", "Rond-point", "Route à 1 chaussée",
                  "Route à 2 chaussées", "Type autoroutier"
                  ])
              )
        ),
        ("train",
          path_train,
          lambda x:
              (x["ETAT"]=="En service")
              &
              (x["POS_SOL"]!=-1)#Avoid tunnels
          )
        ]:
    vector_layer = gpd.read_file(
        path_vector_layer
    )
    vector_layer[f"is_{nom}"]=True

    vector_layer = vector_layer[filters(vector_layer)]#filtering

    ocsge = add_vector_layer(
        ocsge,
        vector_layer,
        {f"length_{nom}": np.sum},
        f"plus_{nom}.gpkg",
        f"plus_{nom}",
        "union",
        {f"length_{nom}": lambda join: length_if_condition(join, f"is_{nom}")}
        )

    ocsge = mean_neighbours_values(ocsge,
                                   quanti_cols=[f"length_{nom}"],
                                   quali_cols=[],
                                   )
    ocsge = ocsge.reset_index()

# ocsge = ocsge_plus_vector_layer
try:
    ocsge = ocsge.drop(columns=["length_routes_mean_1m"])
except:
    print("length_routes_mean_1m n'existait pas")
nom = "routes"
ocsge = mean_neighbours_values(ocsge,
                               quanti_cols=[f"length_{nom}"],
                               quali_cols=[],
                               )
try:
    ocsge = ocsge.reset_index()
except:
    pass

ocsge.to_file(
    f"plus_{nom}.gpkg"
    )

#%%MNS
path_MNS_dalles = "E://MNS_69_dalles.gpkg"
dalles = gpd.read_file(path_MNS_dalles)
# ocsge[["mean_MNS", "std_MNS"]] = ocsge.apply(
#     recuperer_raster_1D_dalle,
#     axis=1,
#     args=[dalles, ["mean_MNS", "std_MNS"]])
# ocsge.to_file(
#     "E://plus_MNS.gpkg"
#     )
i0 = 380000
pas = 10000
for i in range(i0, len(ocsge), pas):
    not_working = True
    while not_working:
        not_working = False
        try:
            print(f"{i}/{len(ocsge)}", datetime.datetime.now())
            ocsge.loc[i:i+pas-1, [
                "mean_MNS", "std_MNS"]
                ] = ocsge.loc[i:i+pas-1].apply(
                    recuperer_raster_1D_dalle,
                    axis=1,
                    args=[dalles, ["mean_MNS", "std_MNS"]]
                    )
            ocsge.to_file(
                f"E://plus_MNS_{i}.gpkg"
                )
        except:
            not_working = True
#%%MNT
path_MNT_dalles = "E://MNT_69_dalles.gpkg"
dalles = gpd.read_file(path_MNT_dalles)
# ocsge[["mean_MNT", "std_MNT"]] = ocsge.apply(
#     recuperer_raster_1D_dalle,
#     axis=1,
#     args=[dalles, ["mean_MNT", "std_MNT"]])
# ocsge.to_file(
#     "E://plus_MNT.gpkg"
#     )
i0 = 390000
pas = 10000
for i in range(i0, len(ocsge), pas):
    not_working = True
    while not_working:
        not_working = False
        try:
            print(f"{i}/{len(ocsge)}", datetime.datetime.now())
            ocsge.loc[i:i+pas-1, [
                "mean_MNT", "std_MNT"]
                ] = ocsge.loc[i:i+pas-1].apply(
                    recuperer_raster_1D_dalle,
                    axis=1,
                    args=[dalles, ["mean_MNT", "std_MNT"]]
                    )
            ocsge.to_file(
                f"E://plus_MNT_{i}.gpkg"
                )
        except Exception as e:
            print(e)
            not_working = True

#%%pentes
path_MNT_dalles = "E://pentes.gpkg"
dalles = gpd.read_file(path_MNT_dalles)
# ocsge[["mean_MNT", "std_MNT"]] = ocsge.apply(
#     recuperer_raster_1D_dalle,
#     axis=1,
#     args=[dalles, ["mean_MNT", "std_MNT"]])
# ocsge.to_file(
#     "E://plus_MNT.gpkg"
#     )
i0 = 0
pas = 10000
for i in range(i0, len(ocsge), pas):
    not_working = True
    while not_working:
        not_working = False
        try:
            print(f"{i}/{len(ocsge)}", datetime.datetime.now())
            ocsge.loc[i:i+pas-1, [
                "mean_slope", "std_slope"]
                ] = ocsge.loc[i:i+pas-1].apply(
                    recuperer_raster_1D_dalle,
                    axis=1,
                    args=[dalles, ["mean_slope", "std_slope"]]
                    )
            ocsge.to_file(
                f"E://plus_slope_{i}.gpkg"
                )
        except Exception as e:
            print(e)
            not_working = True

#%%Foursquare filtering

path_communes_bd_topo = os.path.join(
    path_bd_topo,
    "ADMINISTRATIF",
    "COMMUNE.shp"
    )
communes = gpd.read_file(path_communes_bd_topo)
batiments = gpd.read_file(path_bati_bd_topo)

fsq_path = os.path.join(
    "Documents",
    "Donnees",
    "foursquare",
    "places.gpkg")

fsq = gpd.read_file(
    fsq_path,
    layer="places",
    driver="GPKG"
    )

fsq = filtrer_foursquare(fsq, batiments, communes)

fsq.to_file(
    os.path.join(
        "Documents",
        "Donnees",
        "foursquare",
        "fsq_filtered.gpkg")
    )

#%% évaluation de la qualité du filtrage

fsq_filtered = gpd.read_file(
    os.path.join(
        "Documents",
        "Donnees",
        "foursquare",
        "fsq_filtered.gpkg")
    )

pts_gimont = gpd.read_file(
    os.path.join(
        "Documents",
        "Donnees",
        "foursquare",
        "Points_gimont.gpkg")
    )

pts_gimont = pts_gimont[pts_gimont["Vu"].astype(bool)]

inters = fsq_filtered.overlay(pts_gimont,
                              how="intersection",
                              keep_geom_type=True)

inters = inters[inters["name_1"] == inters["name_2"]]

pts_bons_gardes = np.sum(inters["A la bonne place"].astype(int))
pts_pas_bons_gardes = len(inters) - pts_bons_gardes
pts_bons_rejetes = np.sum(pts_gimont["A la bonne place"]) - pts_bons_gardes
pts_pas_bons_rejetes = np.sum(1-pts_gimont["A la bonne place"]) - pts_pas_bons_gardes

print("Matrice de confusion")
print(pts_bons_gardes, pts_pas_bons_gardes)
print(pts_bons_rejetes, pts_pas_bons_rejetes)

precision = pts_bons_gardes / (pts_bons_gardes + pts_pas_bons_gardes)
rappel = pts_bons_gardes / (pts_bons_gardes + pts_bons_rejetes)
accuracy = (pts_bons_gardes + pts_pas_bons_rejetes) / (
    pts_bons_gardes + pts_pas_bons_rejetes + pts_pas_bons_gardes + pts_bons_rejetes
    )

print("Précision :", precision)
print("Rappel :", rappel)
print("Accuracy :", accuracy)

#%%Link between values and classes

all_texts = link_values_class(ocsge,
                  #quantitative cols
                  ['Mean_snv',
                   'US3',
                   'length_train_mean_1m',
                   'osm surf agricol_mean_1m',
                   'signature_5',
                   'Surf Industriel_mean_1m',
                   'stdB_mean_1m',
                   'stdPIR',
                   'densite',
                   'elongation_mean_1m',
                   'osm surf tertiary',
                   'meanV',
                   'Nb Agricole_mean_1m',
                   'osm nb agricol',
                   'HAUTEUR',
                   'Surf Résidentiel',
                   'Surf Commercial et services',
                   'signature_17',
                   'surface',
                   'osm surf resid',
                   'length_routes',
                   'Nb Religieux_mean_1m',
                   'length_train',
                   'osm_pts_us5_mean_1m',
                   'osm_pts_us2_mean_1m',
                   'Nb Sportif',
                   'signature_13',
                   'osm surf resid_mean_1m',
                   'osm_pts_us3',
                   'signature_1_mean_1m',
                   'stdR',
                   'HAUTEUR_mean_1m',
                   'Nb Commercial et services_mean_1m',
                   'nb_parcell',
                   'osm_pts_us2',
                   'signature_4_mean_1m',
                   'US2_mean_1m',
                   'signature_7_mean_1m',
                   'signature_9_mean_1m',
                   'Nb Religieux',
                   'osm surf industrial_mean_1m',
                   'Nb Sportif_mean_1m',
                   'signature_17_mean_1m',
                   'surface_mean_1m',
                   'signature_8_mean_1m',
                   'signature_11',
                   'signature_3_mean_1m',
                   'Ind',
                   'frac_surf_cimetieres',
                   'osm surf indif_mean_1m',
                   'Surf Sportif',
                   'Nb Annexe_mean_1m',
                   'signature_19_mean_1m',
                   'osm nb resid_mean_1m',
                   'stdPIR_mean_1m',
                   'signature_7',
                   'osm nb industrial',
                   'length_routes_mean_1m',
                   'stdV_mean_1m',
                   'signature_10',
                   'signature_3',
                   'stdV',
                   'convexite_mean_1m',
                   'signature_19',
                   'osm nb agricol_mean_1m',
                   'signature_1',
                   'signature_18',
                   'signature_16_mean_1m',
                   'US5',
                   'Nb Indifférencié',
                   'Log_soc',
                   'US2',
                   'signature_9',
                   'osm nb indif_mean_1m',
                   'signature_11_mean_1m',
                   'osm surf indif',
                   'surf_RPG',
                   'convexite',
                   'Surf Indifférencié_mean_1m',
                   'signature_8',
                   'signature_5_mean_1m',
                   'signature_14_mean_1m',
                   'meanV_mean_1m',
                   'nb_parcell_mean_1m',
                   'signature_16',
                   'signature_15_mean_1m',
                   'Nb Industriel_mean_1m',
                   'Nb Industriel',
                   'Surf Résidentiel_mean_1m',
                   'signature_0_mean_1m',
                   'signature_2_mean_1m',
                   'osm nb indif',
                   'signature_6_mean_1m',
                   'nb_local',
                   'holes',
                   'Mean_snv_mean_1m',
                   'surf_RPG_mean_1m',
                   'za_us3',
                   'signature_15',
                   'US3_mean_1m',
                   'densite_mean_1m',
                   'Surf Agricole',
                   'za_us2',
                   'osm nb tertiary',
                   'nb_local_mean_1m',
                   'compacite_mean_1m',
                   'Nb Résidentiel_mean_1m',
                   'meanB',
                   'meanPIR_mean_1m',
                   'signature_0',
                   'za_us2_mean_1m',
                   'signature_6',
                   'US5_mean_1m',
                   'osm nb resid',
                   'osm nb industrial_mean_1m',
                   'ERP',
                   'meanB_mean_1m',
                   'signature_10_mean_1m',
                   'Nb Annexe',
                   'holes_mean_1m',
                   'meanR_mean_1m',
                   'Surf Industriel',
                   'signature_13_mean_1m',
                   'Surf Sportif_mean_1m',
                   'signature_4',
                   'osm nb tertiary_mean_1m',
                   'Surf Commercial et services_mean_1m',
                   'Surf Religieux_mean_1m',
                   'Surf Annexe_mean_1m',
                   'meanR',
                   'signature_2',
                   'Log_soc_mean_1m',
                   'signature_18_mean_1m',
                   'POPULATION',
                   'Nb Résidentiel',
                   'Surf Religieux',
                   'Ind_mean_1m',
                   'Surf Annexe',
                   'za_us3_mean_1m',
                   'stdR_mean_1m',
                   'osm surf agricol',
                   'Surf Indifférencié',
                   'osm_pts_us3_mean_1m',
                   'signature_14',
                   'elongation',
                   'ERP_mean_1m',
                   'compacite',
                   'signature_12',
                   'Nb Commercial et services',
                   'osm surf tertiary_mean_1m',
                   'POPULATION_mean_1m',
                   'Nb Agricole',
                   'osm surf industrial',
                   'signature_12_mean_1m',
                   'meanPIR',
                   'Nb Indifférencié_mean_1m',
                   'stdB',
                   'Surf Agricole_mean_1m',
                   'osm_pts_us5',
                   'frac_surf_aerodromes'
                   ],
                  #qualitative cols
                  ['TYP_IRIS',
                   'TYP_IRIS_mean_1m',
                   'CODE_18',
                   'CODE_18_mean_1m',
                   'OSO',
                   'OSO_mean_1m',
                   'CODE_CS',
                   'CODE_CS_mean_1m',
                   'usage',
                   'usage_mean_1m'],
                  #unit
                  ['€',
                   'm²',
                   'm',
                   'm²',
                   '/',
                   'm²',
                   '/',
                   '/',
                   'inhabitant/m²',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   'm',
                   'm²',
                   'm²',
                   '/',
                   'm²',
                   'm²',
                   'm',
                   '/',
                   'm',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   'm',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   'Inhabitants',
                   'surface fraction',
                   'm²',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   'm²',
                   'm²',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'holes',
                   '€',
                   'm²',
                   'surface fraction',
                   '/',
                   'm²',
                   'inhabitant/m²',
                   'm²',
                   'surface fraction',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'surface fraction',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'holes',
                   '/',
                   'm²',
                   '/',
                   'm²',
                   '/',
                   '/',
                   'm²',
                   'm²',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   'inhabitants',
                   '/',
                   'm²',
                   'inhabitants',
                   'm²',
                   'surface fraction',
                   '/',
                   'm²',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   'inhabitants',
                   '/',
                   'm²',
                   '/',
                   '/',
                   '/',
                   '/',
                   'm²',
                   '/',
                   'surface fraction']
                  )