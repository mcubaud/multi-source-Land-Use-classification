#%% -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:33:24 2023

@author: MCubaud
"""
import numpy as np

class Source(list):
    def __init__(self, *args, name="source"):
        super().__init__(*args)
        self.name = name 

class Sources():
    
    def __init__(self, objective):
        
        self.Geom_cols = Source(['surface', 'convexite', 'holes',
                     'compacite', 'elongation'
                    ] + [
                        "signature_"+str(i) for i in range(20)
                        ], name="Geometry")

        self.radiometric_cols = Source([
            'meanR', 'meanV', 'meanB', 'meanPIR',
            'stdR', 'stdV', 'stdB', 'stdPIR'], name="Radiometry")

        # self.CS_cols = ['CS1.1.1.2', 'CS1.1.2.1', 'CS2.2.1',
        #            'CS1.1.1.1', 'CS2.1.1.1', 'CS1.2.2',
        #            'CS2.1.1.3', 'CS2.1.3', 'CS2.1.2',
        #            'CS2.1.1.2', 'CS1.1.2.2', 'CS1.2.1',
        #            'code_cs_mean_1m']

        self.CS_cols = Source(
            ['code_cs', 'code_cs_mean_1m'],
            name="OCSGE LC"
            )

        self.BD_TOPO_bati_cols = Source(
            [
            'Surf Agricole', 'Surf Annexe',
            'Surf Commercial et services', 'Surf Sportif',
            'Surf Industriel', 'Surf Religieux', 'Surf Résidentiel',
            'Surf Indifférencié', 'HAUTEUR',
            'Nb Agricole', 'Nb Annexe',
            'Nb Commercial et services', 'Nb Sportif',
            'Nb Industriel', 'Nb Religieux', 'Nb Résidentiel',
            'Nb Indifférencié'],
            name="BD_TOPO_buildings"
            )
        
        self.OSO = Source(["OSO", "OSO_mean_1m"], name="OSO")

        self.CLC = Source(["CODE_12", 'CODE_12_mean_1m'], name="CLC")
        
        if objective == "US235":

            self.OSM_cols = Source([
                'osm surf resid', 'osm surf agricol',
                'osm surf industrial',
                'osm surf tertiary', 'osm surf indif',
                'osm nb resid', 'osm nb agricol',
                'osm nb industrial',
                'osm nb tertiary', 'osm nb indif',
                'osm_pts_us2', 'osm_pts_us3',
                'osm_pts_us5'], name="OSM")

            self.BD_TOPO_autres_cols = Source(
                ["za_us2", "za_us3", "ERP"], 
                name="BD_TOPO_other"
                )

            self.IRIS = Source(
                ["densite", "TYP_IRIS", "POPULATION"],
                name="INSEE"
                )

            self.Foncier = Source(
                ["usage_dgfip", "US2", "US3", "US5"],
                name="Land Files"
                )
            
            
        elif objective == "all_LU":
            
            self.OSM_cols = Source([
                #osm polygons
                'osm_LU1_1_surface',
                'osm_LU1_1_number',
                'osm_LU1_2_surface',
                'osm_LU1_2_number',
                'osm_LU2_surface',
                'osm_LU2_number',
                'osm_LU3_surface',
                'osm_LU3_number',
                'osm_LU4_1_1_surface',
                'osm_LU4_1_1_number',
                'osm_LU4_1_2_surface',
                'osm_LU4_1_2_number',
                'osm_LU4_1_3_surface',
                'osm_LU4_1_3_number',
                'osm_LU6_1_surface',
                'osm_LU6_1_number',
                'osm_LU4_3_surface',
                'osm_LU4_3_number',
                'osm_LU5_surface',
                'osm_LU5_number',
                'osm_LU6_2_surface',
                'osm_LU6_2_number',
                #osm landuse polygons
                'osm_LU1_1_landuse_surface',
                'osm_LU2_landuse_surface',
                'osm_LU3_landuse_surface',
                'osm_LU5_landuse_surface',
                #osm lines
                'osm_LU4_1_1_length',
                'osm_LU4_1_2_length',
                'osm_LU4_1_3_length',
                #osm points
                'pts_LU1_1',
                'pts_LU2',
                'pts_LU3',
                'pts_LU4_1_1',
                'pts_LU4_1_2',
                'pts_LU4_1_3',
                'pts_LU5'
                ],
                name="OSM"
                )
            
            self.BD_TOPO_autres_cols = Source(
                [
                'za_us1_1',
                 'za_us1_3',
                 'za_us1_4',
                 'za_us2',
                 'za_us3',
                 'za_us4_3',
                 "ERP",
                 'frac_surf_hydro',
                 'NATURE_hydro',
                 'frac_surf_aerodromes',
                 'frac_surf_cimetieres',
                 'length_routes',
                 'length_train'
                 ],
                name="BD_TOPO_other")
            
            self.IRIS = Source([
                'POPULATION',
                'Mean_snv',
                'Ind',
                'Log_soc',
                'densite',
                'TYP_IRIS'
                ], name="INSEE")
            
            list_US_Foncier = [
                'US11', 'US12', 'US13',# 'US14', 'US15',
                'US2', 'US3',
                'US411', 'US412', 'US413', 'US414', 'US415', 'US42', 'US43',
                'US5', 'US61', 'US62', 'US63', 'US66'
                    ]
            
            self.Foncier = Source([
                f"land_files_{US.replace('US', 'LU')}_area"
                for US in list_US_Foncier
                ], name="Land Files")
            self.Foncier += [
                "usage_dgfip"
                ]
            
            
            self.RPG = Source(['surf_RPG', 'surf_RPG_mean_1m'], name="RPG")

        #Add the neighboring attributes not previously listed
        self.Geom_cols += [col+'_mean_1m' for col in self.Geom_cols]
        self.radiometric_cols += [col+'_mean_1m' for col in self.radiometric_cols]
        self.OSM_cols += [col+'_mean_1m' for col in self.OSM_cols]
        self.BD_TOPO_bati_cols += [col+'_mean_1m' for col in self.BD_TOPO_bati_cols]
        self.BD_TOPO_autres_cols += [col+'_mean_1m' for col in self.BD_TOPO_autres_cols]
        self.IRIS += [col+'_mean_1m' for col in self.IRIS]
        self.Foncier += [col+'_mean_1m' for col in self.Foncier]

        self.liste_sources = [
            self.Geom_cols,
            self.radiometric_cols,
            self.CS_cols,
            self.CLC,
            self.OSO,
            self.BD_TOPO_bati_cols,
            self.BD_TOPO_autres_cols,
            self.IRIS,
            self.Foncier,
            self.OSM_cols,
            ]

        self.noms_sources = [
            "Geometry",
            "Radiometry",
            "OCSGE LC",
            "CLC",
            "OSO",
            "BD_TOPO_buildings",
            "BD_TOPO_other",
            "INSEE",
            "Land Files",
            "OSM",
            ]


        self.categorical_cols_list = [
            "TYP_IRIS",
            "TYP_IRIS_mean_1m",
            "usage",
            "usage_dgfip",
            "usage_dgfip_mean_1m"
            ] + self.CLC + self.OSO + self.CS_cols
        
        if objective == "all_LU":
            self.liste_sources.append(self.RPG)
            self.noms_sources.append("RPG")
            self.categorical_cols_list.append("NATURE_hydro")


        self.all_cols = list(
            np.concatenate(self.liste_sources)
            )

        self.not_m1_cols = [col
                    for col in self.all_cols
                    if '_mean_1m' not in col]
        
        self.not_Geom_cols = list(set(self.all_cols) - set(self.Geom_cols))
        self.not_radiometric_cols = list(set(self.all_cols) - set(self.radiometric_cols))
        self.not_CS_cols = list(set(self.all_cols) - set(self.CS_cols))
        self.not_OSM_cols = list(set(self.all_cols) - set(self.OSM_cols))
        self.not_BD_TOPO_bati_cols = list(set(self.all_cols) - set(self.BD_TOPO_bati_cols))
        self.not_BD_TOPO_autres_cols = list(set(self.all_cols) - set(self.BD_TOPO_autres_cols))
        self.not_IRIS = list(set(self.all_cols) - set(self.IRIS))
        self.not_OSO = list(set(self.all_cols) - set(self.OSO))
        self.not_CLC = list(set(self.all_cols) - set(self.CLC))
        self.not_Foncier = list(set(self.all_cols) - set(self.Foncier))
        self.not_CLC_neither_CS_cols = list(set(self.all_cols) - set(self.CLC) - set(self.CS_cols))
        
        self.list_not_source = [
            self.not_Geom_cols ,
            self.not_radiometric_cols,
            self.not_CS_cols,
            self.not_CLC,
            self.not_OSO,
            self.not_BD_TOPO_bati_cols,
            self.not_BD_TOPO_autres_cols,
            self.not_IRIS,
            self.not_Foncier,
            self.not_OSM_cols,
            self.not_CLC_neither_CS_cols,
            ]
        
        self.not_source_names = [
            "Not geometry",
            "Not radiometry",
            "Not OCSGE LC",
            "Not CLC",
            "Not OSO",
            "Not BD_TOPO_buildings",
            "Not BD_TOPO_other",
            "Not INSEE",
            "Not Land Files",
            "Not OSM",
            "Not CLC - OCSGE LC"
            ]
        
        if objective == "all_LU":
            self.not_RPG = list(set(self.all_cols) - set(self.RPG))
            self.list_not_source.append(self.not_RPG)
            self.not_source_names.append("not RPG")
            
            
        self.dict_attributes_sources ={}
        for i in range(len(self.liste_sources)):
            source = self.liste_sources[i]
            for j in range(len(source)):
                self.dict_attributes_sources[source[j]] = self.noms_sources[i]
                
                
                
                
    def find_source_name(self, list_attributes):
        set_attributes = set(list_attributes)
        for i, source in enumerate(self.liste_sources):
            if len(set_attributes.symmetric_difference(source))==0:
                return self.noms_sources[i]
        for i, source in enumerate(self.list_not_source):
            if len(set_attributes.symmetric_difference(source))==0:
                return self.not_source_names[i]
        set_attributes_plus_cs = set_attributes.union(self.CS_cols)
        for i, source in enumerate(self.liste_sources):
            if len(set_attributes_plus_cs.symmetric_difference(source))==0:
                return self.noms_sources[i]+" - OCSGE LC"
        for i, source in enumerate(self.list_not_source):
            if len(set_attributes_plus_cs.symmetric_difference(source))==0:
                return self.not_source_names[i]+" - OCSGE LC"
        if len(set_attributes.symmetric_difference(self.not_m1_cols))==0:
               return "Without neighbors means"
        if len(set_attributes.union(["code_cs"]).symmetric_difference(self.not_m1_cols))==0:
               return "Without neighbors means - OCSGE LC"
        list_sources = []
        for i, source in enumerate(self.liste_sources):
            if len(set(source).difference(set_attributes))==0:
                # this source is in the list
                list_sources.append(self.noms_sources[i])
        if len(list_sources)!=0:
            return "Not (" + "+".join( set(self.noms_sources).difference(list_sources) )+")"
        else:
            return "source not identified"
            
# %%
