# Multi-source Land-Use classification 🌳🏙️
This repository contains a Python script that implements a data-fusion based method for land use classification. The code is based on research conducted for the article:
> Cubaud, M., Le Bris, A., Jolivet, L., & Olteanu-Raimond, A. M. (2024). Assessing the transferability of a multi-source land use classification workflow across two heterogeneous urban and rural areas. International Journal of Digital Earth, 17(1). https://doi.org/10.1080/17538947.2024.2376274

Additionally, the methodologies employed in this code are highly related to our previous work:
 
> Cubaud, M., Le Bris, A., Jolivet, L., and Olteanu-Raimond, A.-M. (2023). Comparison of Two Data Fusion Approaches for Land Use Classification. International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48(1/W2), 699–706. DOI: 10.5194/isprs-archives-XLVIII-1-W2-2023-699-2023.

## Objectives 🎯

The research evaluates a multi-modal Machine Learning framework applied to LU classification in French departments, aiming to understand model transferability challenges and the contributions of various data sources.

## Required data 📊
The data on which the code is applied is available in the following Zenodo repository:

> Cubaud, M., LE BRIS, A., Jolivet, L., Olteanu-Raimond, A.-M., & Institut national de l'information géographique et forestière. (2024). OCS GE Land Use dataset including the French departments of Gers and Rhône. Zenodo. https://doi.org/10.5281/zenodo.10462844

## Files 📁
- **land_use_classification_pipeline.py**: Main file, it defines the workflow for LU classification and applies it to the OCS GE Land Use dataset of Rhône or Gers
- **from_one_dep_to_another.py**: File in which the workflow is trained in one study area, and applied to the other. 
- **definition_of_sources.py**: File to define the sources.
- **plot_function.py**: File with visualization functions.
- **color_f_US.py**: File which assigns to each LU class a color.
- **attributes_extraction_from_the_sources.py**: File to create the dataset from the sources. Link to the sources are provided in the Data sources section
- **OSM_to_OCSGE.csv**: File to describe how to attribute to some OSM points, lines and polygons a OCS GE Land Use class.


## Warning ⚠️

Please note that the source for Land Files is not directly available due to privacy reasons. While the other data sources are accessible through the provided links, the Land Files cannot be accessed directly from this repository. We apologize for any inconvenience this may cause. Attributes from the Land Files could not have been shared in the zenodo repository neither. The results obtained in the article are thus not fully reproducible. The code still works without these attributes by explicitly removing them.

## Data sources 🌐
The dataset including the LU polygons of both study areas and the constructed attributes is available at https://doi.org/10.5281/zenodo.10462844
- BD ORTHO is available at https://geoservices.ign.fr/bdortho.
- CLC is available at https://land.copernicus.eu/en/products/corine-land-cover.
- OSO is available at https://www.theia-land.fr/product/carte-doccupation-des-sols-de-la-france-metropolitaine/.
- BD TOPO is available at https://geoservices.ign.fr/bdtopo.
- RPG is available at https://geoservices.ign.fr/rpg.
- INSEE's IRIS polygons are available at https://geoservices.ign.fr/irisge, while their statistical surveys are at https://www.insee.fr/fr/statistiques/7704076. INSEE grid data is available at https://www.insee.fr/fr/statistiques/6215138?sommaire=6215217.
- OpenStreetMap data was downloaded using https://download.openstreetmap.fr/extracts/europe/france/.
- Land Files are not directly available due to protection of the privacy reasons.
