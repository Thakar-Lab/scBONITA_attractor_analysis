############################
##### Import libraries #####
############################
from cmath import nanj
from copyreg import pickle
import numpy as np
import pandas as pd
import os
import os.path
from os import path
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np
from ast import literal_eval
from random import seed
import random
import pickle
import bioservices
import math
from scipy import stats
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
import joypy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time
import pystablemotifs as sm
import pyboolnet
import pystablemotifs.export as ex
from scipy.stats import chi2_contingency

seed(15) #set seed for reproducibility


#####################################
##### Create folders for output #####
#####################################
if os.path.isdir('Clustering_Output'):
    print("Clustering_Output File Already Exists")
else:
    os.mkdir('Clustering_Output')

if os.path.isdir('ERS_Size_Output'):
    print("ERS_Size_Output File Already Exists")
else:
    os.mkdir('ERS_Size_Output')
    
if os.path.isdir('TSNE_Output'):
    print("TSNE_Output File Already Exists")
else:
    os.mkdir('TSNE_Output')
    
if os.path.isdir('Cell_Type_Analysis_Output'):
    print("Cell_Type_Analysis_Output File Already Exists")
else:
    os.mkdir('Cell_Type_Analysis_Output')

if os.path.isdir('Heatmap_Output'):
    print("Heatmap_Output File Already Exists")
else:
    os.mkdir('Heatmap_Output')
    
if os.path.isdir('attractors_per_dtst_ntrk'):
    print("attractors_per_dtst_ntrk File Already Exists")
else:
    os.mkdir('attractors_per_dtst_ntrk')
    
if os.path.isdir('Gene_Expression'):
    print("Gene_Expression File Already Exists")
else:
    os.mkdir('Gene_Expression')
    
if os.path.isdir('Cells_Names_Per_Attractors'):
    print("Cells_Names_Per_Attractors File Already Exists")
else:
    os.mkdir('Cells_Names_Per_Attractors')
    
if os.path.isdir('criteria_cells_figures'):
    print("criteria_cells_figures File Already Exists")
else:
    os.mkdir('criteria_cells_figures')
    
    
if os.path.isdir('cont_disc_comparison_figs'):
    print("cont_disc_comparison_figs File Already Exists")
else:
    os.mkdir('cont_disc_comparison_figs')
    
    
#########################
##### All functions #####
#########################

############################################
##### Function that generated Figure 1 #####
############################################
def Check_Gene_Expression(cells_criteria_list,List_Of_Addresss_To_Datasets,List_of_Addresses_To_Azimuth_Predictions,Datasets_Labels,all_dtsts_colors_map,gene_expression_saving_path,azimuth_used=True,plot_size=(10,10)):
    cells_genes_expression_for_all_datasets = {}
    for dtst_addrss_idx in range(len(List_Of_Addresss_To_Datasets)):
        print("Importing " + Datasets_Labels[dtst_addrss_idx])
        raw_csv_df = pd.read_csv(List_Of_Addresss_To_Datasets[dtst_addrss_idx],index_col=0)
        print("Done Importing " + Datasets_Labels[dtst_addrss_idx])
        genes_names_in_df = list(raw_csv_df.index.values)
        cells_names_in_df = list(raw_csv_df.columns)
        num_cells = len(cells_names_in_df)
        cells_count_with_genes_on = {}
        if azimuth_used == True:
            azimuth_df = pd.read_csv(List_of_Addresses_To_Azimuth_Predictions[dtst_addrss_idx],index_col=0)
            azimuth_cells = list(azimuth_df.index)
            azimuth_type_pre = list(azimuth_df["type"])
            azimuth_type = []
            for i in azimuth_type_pre:
                if i == "transitional stage B cell":
                    azimuth_type.append("transitional\nstage B cell")
                elif i == "naive B cell":
                    azimuth_type.append("naive\nB cell")
                elif i == "classical memory B cell":
                    azimuth_type.append("classical\nmemory B cell")
                elif i == "IgM memory B cell":
                    azimuth_type.append("IgM memory\nB cell")
                elif i == "double negative memory B cell":
                    azimuth_type.append("double negative\nmemory B cell")
                else:
                    print("NOT FOUND!!!")
                    print(i)
                    
            azimuth_dict = {}
            for idx in range(len(azimuth_cells)):
                azimuth_dict[azimuth_cells[idx]] = azimuth_type[idx]
        
        for each_cell_typ in cells_criteria_list:
            cells_count_with_genes_on[each_cell_typ] = 0
            
        if azimuth_used == False:
            cells_count_with_genes_on["Other Cells"] = 0
        typed_cells = []
            
        for cell_type in cells_criteria_list:
            print("Calculating for cell type " + str(cell_type))
            #store this cell type criteria
            criteria_for_this_cell_type = cells_criteria_list[cell_type]
            #loop through each cell in this dataset and test if it meets criteria
            for each_cell_name_idx in range(len(cells_names_in_df)):
                if azimuth_used == False:
                    genes_on_criteria_met = 1
                    genes_off_criteria_met = 1
                    #test genes on
                    if len(criteria_for_this_cell_type["on"]) > 0:
                        on_genes = criteria_for_this_cell_type["on"]
                        for on_gn in on_genes:
                            if on_gn not in genes_names_in_df:
                                if each_cell_name_idx == 0:
                                    print("Gene " + on_gn + " NOT Found in " + Datasets_Labels[dtst_addrss_idx])
                                genes_on_criteria_met = 0
                                continue
                            if raw_csv_df.loc[on_gn,cells_names_in_df[each_cell_name_idx]] > 0.001:
                                continue
                            else:
                                genes_on_criteria_met = 0
                                break
                    #skip checking off genes if on genes criteria already was not met
                    if genes_on_criteria_met == 0:
                        continue
                    #test genes off
                    if len(criteria_for_this_cell_type["off"]) > 0:
                        off_genes = criteria_for_this_cell_type["off"]
                        for off_gn in off_genes:
                            if off_gn not in genes_names_in_df:
                                if each_cell_name_idx == 0:
                                    print("Gene " + off_gn + " NOT Found in " + Datasets_Labels[dtst_addrss_idx])
                                genes_off_criteria_met = 0
                                continue
                            if raw_csv_df.loc[off_gn,cells_names_in_df[each_cell_name_idx]] <= 0.001:
                                continue
                            else:
                                genes_off_criteria_met = 0
                                break
                    #if both criteria are met then this cell's info should be added to percs and dist dictionaries
                    if (genes_on_criteria_met == 1) and (genes_off_criteria_met == 1):
                        cells_count_with_genes_on[cell_type] += 1
                        typed_cells.append(cells_names_in_df[each_cell_name_idx])
                elif azimuth_used == True:
                    if azimuth_dict[cells_names_in_df[each_cell_name_idx]] == cell_type:
                        cells_count_with_genes_on[cell_type] += 1
                        typed_cells.append(cells_names_in_df[each_cell_name_idx])
                print("Tested Critetira for Cell #" + str(each_cell_name_idx+1) + " Out Of " + str(len(cells_names_in_df)) + " Cells", end = "\r")
            print("\n")
                
            #convert counts to percentages
            cells_count_with_genes_on[cell_type] = (cells_count_with_genes_on[cell_type]/num_cells)*100
        print("\n\n")
        
        #for other cells
        if azimuth_used == False:
            for each_cl in cells_names_in_df:
                if each_cl not in typed_cells:
                    cells_count_with_genes_on["Other Cells"] += 1
            cells_count_with_genes_on["Other Cells"] = (cells_count_with_genes_on["Other Cells"]/num_cells)*100
        
        
        cells_genes_expression_for_all_datasets[Datasets_Labels[dtst_addrss_idx]] = cells_count_with_genes_on


    print(cells_genes_expression_for_all_datasets)
    
    
    ####### NEW CODE ########
    List_Datasets_Names = [d for d in Datasets_Labels]
    List_Datasets_Names_Fixed = []
    for dtst_nm in List_Datasets_Names:
        if dtst_nm == "HIV_dataset":
            List_Datasets_Names_Fixed.append("HIV Dataset")
        elif dtst_nm == "Lung_Cancer_dataset":
            List_Datasets_Names_Fixed.append("Lung Cancer\nDataset")
        elif dtst_nm == "Breast_Cancer_dataset":
            List_Datasets_Names_Fixed.append("Breast Cancer\nDataset")
        elif dtst_nm == "Mild_Severe_Covid_dataset":
            List_Datasets_Names_Fixed.append("Mild-Severe\nCOVID Dataset")
        elif dtst_nm == "Severe_Covid_dataset":
            List_Datasets_Names_Fixed.append("Severe COVID\nDataset")
    List_Cells_Types_Names = [c for c in cells_criteria_list]
    if azimuth_used == False:
        List_Cells_Types_Names = List_Cells_Types_Names + ["Other Cells"]
    X_axis = np.array(list(range(0,(2*len(List_Datasets_Names)),2)))
    plt.figure(figsize=plot_size)
    ntrk_ctr = 0
    for ech_cl_type in List_Cells_Types_Names:
        crnt_celltype_ordered_genes_cells_percs = []
        for ech_dtst in List_Datasets_Names:
            crnt_celltype_ordered_genes_cells_percs.append(cells_genes_expression_for_all_datasets[ech_dtst][ech_cl_type])
        print(list(X_axis+(ntrk_ctr*0.3)))
        print(crnt_celltype_ordered_genes_cells_percs)
        plt.barh(list(X_axis+(ntrk_ctr*0.3)), crnt_celltype_ordered_genes_cells_percs, 0.3, color=all_dtsts_colors_map[ech_cl_type], label=ech_cl_type)
        ntrk_ctr += 1
    #plt.xticks(np.arange(0,len(List_Datasets_Names)*2,2)+((len(List_Cells_Types_Names)/2)*0.3), List_Datasets_Names, fontsize=40)
    plt.yticks(np.arange(0,len(List_Datasets_Names)*2,2)+((len(List_Cells_Types_Names)/2)*0.3), List_Datasets_Names_Fixed, fontsize=40)
    #plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    #plt.grid(color='gray',axis='y',alpha=0.4)
    plt.grid(color='gray',axis='x',alpha=0.4)
    plt.legend(fontsize='x-large')
    #plt.ylabel("Percentage\nof cells", fontsize=60)
    plt.xlabel("Percentage\nof cells", fontsize=40)
    plt.rc('legend',fontsize=100)
    plt.savefig(gene_expression_saving_path + "All_Datasets_Genes_Expression_Cells_Percs.png")
    plt.close()


