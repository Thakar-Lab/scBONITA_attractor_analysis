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
from matplotlib.patches import Patch
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
from matplotlib import rcParams
seed(15) #set seed for reproducibility
print("Done importing libraries")


#########################
##### All functions #####
#########################

#################################################################################################
##### Function that generated barplot for the percentage of B cell subtypes in each dataset #####
#################################################################################################
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
    plt.yticks(np.arange(0,len(List_Datasets_Names)*2,2)+((len(List_Cells_Types_Names)/2)*0.3), List_Datasets_Names_Fixed, fontsize=5)
    #plt.yticks(fontsize=40)
    plt.xticks(fontsize=5)
    #plt.grid(color='gray',axis='y',alpha=0.4)
    plt.grid(color='gray',axis='x',alpha=0.4)
    #plt.legend(fontsize='x-large')
    #plt.ylabel("Percentage\nof cells", fontsize=60)
    plt.xlabel("Percentage\nof cells", fontsize=5)
    #plt.rc('legend',fontsize=100)
    # End of disabling texts around figure
    plt.savefig(gene_expression_saving_path + "All_Datasets_Genes_Expression_Cells_Percs.pdf")
    plt.close()





################################################################################################################
##### Function that generated umap figures for cells in each dataset and the B cell subtype they mapped to #####
################################################################################################################

def umap_by_cell_type(embeddings_addresses,predictions_addresses,hue_dict,datasets_labels,alphas,legend="auto",figure_dims=(10,10),alpha=1,size=1):
    global_df = []
    for dtst_idx in range(len(datasets_labels)):
        embeddings_df = pd.read_csv(embeddings_addresses[dtst_idx])
        predictions_df = pd.read_csv(predictions_addresses[dtst_idx])
        merged_df = pd.merge(embeddings_df,predictions_df,how="inner",on=["cells"])
        print(set(list(merged_df["predictions"])))
        merged_df["dataset"] = [datasets_labels[dtst_idx]]*len(list(merged_df["cells"]))
        if dtst_idx == 0:
            global_df = merged_df
        else:
            global_df = pd.concat([global_df,merged_df],axis=0,ignore_index=True)
    print(global_df)
    print(set(list(global_df["predictions"])))
    fig = plt.figure(figsize=figure_dims)
    sns.scatterplot(data=global_df,x="UMAP_1",y="UMAP_2",hue="dataset",style="predictions",legend=legend,alpha=alpha,size=size)
    plt.savefig("New_Azimuth_Output/scatter_plot.pdf")
    plt.clf()
    for dtst_idx in range(len(datasets_labels)):
        fig = plt.figure(figsize=figure_dims)
        curnt_data = global_df[global_df["dataset"]==datasets_labels[dtst_idx]]
        
        for cell_prdctn in alphas:
            layer = curnt_data[curnt_data["predictions"]==cell_prdctn]
            sns.scatterplot(data=layer,x="UMAP_1",y="UMAP_2",hue="predictions",alpha=alphas[cell_prdctn],size=size, palette={cell_prdctn:hue_dict[cell_prdctn]})
            
        plt.legend([],[], frameon=False)
        plt.xlabel("UMAP 1", fontsize=5, labelpad=4)
        plt.ylabel("UMAP 2", fontsize=5, labelpad=0.1)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.savefig("New_Azimuth_Output/"+datasets_labels[dtst_idx]+"_scatter_plot.pdf")
        plt.clf()
        
        
########################################################################################
##### Function that generated the figure of mean ERS size per dataset-network pair #####
########################################################################################


def Create_ERS_Figures(addresses_to_datasets_files, datasets_labels, pathways_ids, networks_names, ERS_saving_path, Mean_ERS_Sizes_Together=True,errbar_w = 1, errcap_s = 3, fig_2_size=(40,50)):

    all_datasets_in3_ERS_dfs = {}
    for dataset_indx in range(len(addresses_to_datasets_files)):
        all_networks_ERS_dfs_holder = []
        for network_id in pathways_ids:
            imp_scr_address = addresses_to_datasets_files[dataset_indx] + network_id + "_processed.graphml_importanceScores.csv"
            if os.path.exists(imp_scr_address):
                imp_scr_df = pd.read_csv(imp_scr_address, index_col=0)
                ERS_size_df = imp_scr_df[["ObsERS","MaxERS"]]
                ERS_size_df["NetworkName"] = [networks_names[network_id]] * len((ERS_size_df["ObsERS"].tolist()))
                all_networks_ERS_dfs_holder.append(ERS_size_df)
            else:
                print("Pathway " + networks_names[network_id] + " Could Not Be Found!")
        Complete_ERS_df = pd.concat([all_networks_ERS_dfs_holder[0],all_networks_ERS_dfs_holder[1]])
        for ERS_df_indx in range(2,len(all_networks_ERS_dfs_holder)):
            Complete_ERS_df = pd.concat([Complete_ERS_df, all_networks_ERS_dfs_holder[ERS_df_indx]])
        Complete_ERS_df_in2 = Complete_ERS_df[Complete_ERS_df["MaxERS"]==7]
        Complete_ERS_df_in3 = Complete_ERS_df[Complete_ERS_df["MaxERS"]==127]
        all_datasets_in3_ERS_dfs[datasets_labels[dataset_indx]] = Complete_ERS_df_in3

    if Mean_ERS_Sizes_Together == True:
        plt.figure(figsize=fig_2_size)
        #Pathway with lowest mean ERS size for in3
        rcParams.update({'figure.autolayout': False})
        ntrks_ERS_mean_across_dtsts = {}
        ntrks_ERS_sem_across_dtsts = {}
        for dataset_label in datasets_labels:
            dtst_in3_df = all_datasets_in3_ERS_dfs[dataset_label]
            names_of_found_networks = list(set(list(dtst_in3_df["NetworkName"])))
            ERS_means_of_found_networks = []
            ERS_sems_of_found_networks = []
            for ntrk in names_of_found_networks:
                ntrk_df_holder = dtst_in3_df[dtst_in3_df["NetworkName"]==ntrk]
                ntrk_ERS_mean = np.nanmean(np.array(list(ntrk_df_holder["ObsERS"])))
                #making it log2(ERS+1)
                #ntrk_ERS_mean_pre = np.nanmean(np.array(list(ntrk_df_holder["ObsERS"])))
                #ntrk_ERS_mean = math.log2(ntrk_ERS_mean_pre+1)
                
                ntrk_ERS_std = np.nanstd(np.array(list(ntrk_df_holder["ObsERS"])))
                ntrk_ERS_sem = ntrk_ERS_std/(np.sqrt(len(list(ntrk_df_holder["ObsERS"]))))
                #making it log2(ERS+1)
                #ntrk_ERS_sem_pre = ntrk_ERS_std/(np.sqrt(len(list(ntrk_df_holder["ObsERS"]))))
                #ntrk_ERS_sem = 
                
                ERS_means_of_found_networks.append(ntrk_ERS_mean)
                ERS_sems_of_found_networks.append(ntrk_ERS_sem)
            min_indx = ERS_means_of_found_networks.index(min(ERS_means_of_found_networks))
            Chosen_ntrk = names_of_found_networks[min_indx]
            print("For " + dataset_label)
            print("The network with the minimal mean of ERS size is:")
            print(Chosen_ntrk)
            for ntrk_name_indx in range(len(names_of_found_networks)):
                if names_of_found_networks[ntrk_name_indx] not in ntrks_ERS_mean_across_dtsts:
                    ntrks_ERS_mean_across_dtsts[names_of_found_networks[ntrk_name_indx]] = {}
                ntrks_ERS_mean_across_dtsts[names_of_found_networks[ntrk_name_indx]][dataset_label] = ERS_means_of_found_networks[ntrk_name_indx]
                if names_of_found_networks[ntrk_name_indx] not in ntrks_ERS_sem_across_dtsts:
                    ntrks_ERS_sem_across_dtsts[names_of_found_networks[ntrk_name_indx]] = {}
                ntrks_ERS_sem_across_dtsts[names_of_found_networks[ntrk_name_indx]][dataset_label] = ERS_sems_of_found_networks[ntrk_name_indx] 
        
        all_ntrks = []
        ntrk_counter = 0
        ntrk_dtst_counter = 0
        for ntrk_name in ntrks_ERS_mean_across_dtsts:
            all_ntrks.append(ntrk_name)
            for each_dtst in ntrks_ERS_mean_across_dtsts[ntrk_name]:
                dtst_ntrk_mean = ntrks_ERS_mean_across_dtsts[ntrk_name][each_dtst]
                dtst_ntrk_sem = ntrks_ERS_sem_across_dtsts[ntrk_name][each_dtst]
                dtst_color = all_dtsts_colors_map[each_dtst]
                if ntrk_counter == 0:
                    plt.barh(ntrk_counter+(0.3*ntrk_dtst_counter),[dtst_ntrk_mean],0.3,color=dtst_color,label=each_dtst,xerr=[dtst_ntrk_sem],align="center", ecolor='#353839', capsize=errcap_s, error_kw=dict(lw=errbar_w, capsize=errcap_s, capthick=errbar_w), alpha=0.9)
                else:
                    plt.barh(ntrk_counter+(0.3*ntrk_dtst_counter),[dtst_ntrk_mean],0.3,color=dtst_color,xerr=[dtst_ntrk_sem],align="center", ecolor='#353839', capsize=errcap_s, error_kw=dict(lw=errbar_w, capsize=errcap_s, capthick=errbar_w), alpha=0.9)
                ntrk_dtst_counter += 1
            ntrk_dtst_counter = 0
            ntrk_counter += 2
            
        plt.grid(color='gray',axis='x',alpha=0.3)
        plt.yticks(np.arange(0,len(all_ntrks)*2,2)+((len(datasets_labels)/2)*0.3), all_ntrks, fontsize=8)
        
        ##### limit x axis lower range #####
        plt.xlim(left=50)
        plt.xticks(list(np.arange(50,120,20))+[127],fontsize=8)
        
        plt.xlabel("Mean ERS Size", fontsize=8, labelpad=5)
        #plt.ylabel("Networks", fontsize=30, labelpad=10)
        #plt.legend(framealpha=1)
        #vertical lines at 1, 7, 127
        #plt.axvline(x = 1, color = 'black',linestyle="--")
        #plt.axvline(x = 7, color = 'black',linestyle="--")
        #plt.axvline(x = 127, color = 'black',linestyle="--")
        plt.tight_layout()
        plt.savefig(ERS_saving_path + "All_Datasets_ERS_means_BarPlot.pdf")
        plt.close()

        
###############################################################################################################
##### Four functions that generated bar and box plots of how attractors mapped to B cell source/phenotype #####
###############################################################################################################

##########################
##### First function #####
##########################

def Import_Attractors_Mapped_To_By_Cells_And_Their_Counts(addresses_to_datasets_files, pathways_ids,datasets_labels,networks_names,active_gene_tresh=1):

    all_attractors_mapped_to_by_cells = {} #store the list of int formated mapped attractors (values) per dataset and network (keys where dataset and network name separated by "__")
    counts_for_attractors_mapped_to_by_cells = {} #store the dictionaries of attractors_cells_count (values) per dataset and network (keys where dataset and network name separated by "__")
    loaded_dist_dfs = {} #store the loaded distances df (values) per dataset and network (keys where dataset and network name separated by "__")
    datasets_addresses = {} #store the address of the imported df (values) per dataset and network (keys where dataset and network name separated by "__")

    for scBonita_dataset_idx in range(len(addresses_to_datasets_files)): #go through address indx of each dataset
        for pathway_idx in range(len(pathways_ids)): #go through each network's id indx
            attractors_cells_count = {} #stores string format of each attractor (keys) and the values being the number of cells maping to each of these attractors 
            #generate address of cells and attractors distances for this dataset
            distances_file_address = addresses_to_datasets_files[scBonita_dataset_idx] + pathways_ids[pathway_idx] + "_processed.graphml_attractorDistance.csv"

            if os.path.exists(distances_file_address): #check that the dataset's csv file exists
                dist_df = pd.read_csv(distances_file_address, index_col=0) #import distances file
                mapped_attracs_idxs = dist_df["decider"].tolist() #store the indices of the attractors mapped to by cells
                mapped_attracs = [] #stores the int format of attractors mapped to by cells
                for attrac_indx in mapped_attracs_idxs: #go through each attractor indx mapped to by cells (reptition allowed)
                    attractor_based_on_indx = dist_df.columns[attrac_indx] #find the 0/1s pattern for each attractor
                    attrac_holder = list(map(int,(attractor_based_on_indx.strip("[]").split(", ")))) #fix format from strings to ints
                    #make sure attractor isn't all zeros
                    if sum(attrac_holder)<active_gene_tresh:
                        continue
                    #increase the count of cells per attractor
                    if attractor_based_on_indx in attractors_cells_count:
                        attractors_cells_count[attractor_based_on_indx] += 1
                    else:
                        attractors_cells_count[attractor_based_on_indx] = 1
                #go through each mapped attractor and appned it to the list mapped_attracs
                for attract in attractors_cells_count:
                    mapped_attracs.append(attract)
                mapped_attracs = [list(map(int,(k.strip("[]").split(", ")))) for k in mapped_attracs] #fix format of attractors from ints to strings

                all_attractors_mapped_to_by_cells[datasets_labels[scBonita_dataset_idx]+"__"+networks_names[pathways_ids[pathway_idx]]] = mapped_attracs #store the list of int formated mapped attractors (values) per dataset and network (keys where dataset and network name separated by "__")
                counts_for_attractors_mapped_to_by_cells[datasets_labels[scBonita_dataset_idx]+"__"+networks_names[pathways_ids[pathway_idx]]] = attractors_cells_count #store the dictionaries of attractors_cells_count (values) per dataset and network (keys where dataset and network name separated by "__")
                loaded_dist_dfs[datasets_labels[scBonita_dataset_idx]+"__"+networks_names[pathways_ids[pathway_idx]]] = dist_df #store the loaded distances df (values) per dataset and network (keys where dataset and network name separated by "__")
                datasets_addresses[datasets_labels[scBonita_dataset_idx]+"__"+networks_names[pathways_ids[pathway_idx]]] = addresses_to_datasets_files[scBonita_dataset_idx] #store the address of the imported df (values) per dataset and network (keys where dataset and network name separated by "__")
                print("Done Importing attractors mapped to by cells for "+datasets_labels[scBonita_dataset_idx]+"__"+networks_names[pathways_ids[pathway_idx]])
            else:
                print("Pathway " + networks_names[pathways_ids[pathway_idx]] + " Could Not Be Found!")

    return all_attractors_mapped_to_by_cells, counts_for_attractors_mapped_to_by_cells, loaded_dist_dfs, datasets_addresses

###########################
##### Second function #####
###########################

def Attractors_Clustering(all_attractors_mapped_to_by_cells, distance_threshold, dendrogram_saving_pathway, lastp_for_truncate, save_truncated_dendrogram=False, save_dendrogram = True):
    

    set_of_clusters_per_dataset = {} #a dict of dict. outer dict includes keys which are dataset__network. inner dict stores cluster id as keys and values being list of lists. The inner lists being the attractors within that cluster

    for dataset_pathway in all_attractors_mapped_to_by_cells: #go through each dataset_network pair where keys are dataset and network name separated by "__". Values are list of int formated attractors mapped to by cells
        if len(all_attractors_mapped_to_by_cells[dataset_pathway]) == 1: #check if there is only one attractor in list for that dataset_network pair
            #if only one attractor per dataset_network
            set_of_clusters_per_dataset[dataset_pathway] = {1:[]}
            set_of_clusters_per_dataset[dataset_pathway][1].append(all_attractors_mapped_to_by_cells[dataset_pathway][0])
            continue
        #Calculate Distance Matrix for attractors mapped to by cells
        DistanceMatrix_For_Clustering = []
        for attr1_indx in range(len(all_attractors_mapped_to_by_cells[dataset_pathway])): #first go through indices of int formmated mapped attractors per dataset_network pair
            attr1_distances = [] #stores hamming distances between first attractor and the rest of attractors
            for attr2 in all_attractors_mapped_to_by_cells[dataset_pathway]: #second go through of int formmated mapped attractors per dataset_network pair
                attr1_distances.append(sum([abs(i1 - i2) for i1, i2 in zip(all_attractors_mapped_to_by_cells[dataset_pathway][attr1_indx], attr2)])) #Calculates Hamming Distance
            DistanceMatrix_For_Clustering.append(attr1_distances)
            print("Calculated Distances For Attractor #" + str(attr1_indx+1) + " Out Of " + str(len(all_attractors_mapped_to_by_cells[dataset_pathway])) + " Attractors", end = "\r")
        print("Calculated Distances For Attractor #" + str(len(all_attractors_mapped_to_by_cells[dataset_pathway])) + " Out Of " + str(len(all_attractors_mapped_to_by_cells[dataset_pathway])) + " Attractors")
        print("Done calculating Distance Matrix for " + dataset_pathway)

        
        #normalize the distance treshold by the number of nodes in the network
        num_nodes_in_ntrk = len(all_attractors_mapped_to_by_cells[dataset_pathway][0])
        normalized_distance_threshold = int((distance_threshold/100)*num_nodes_in_ntrk)
        print("Normalized distance treshold for " + dataset_pathway + ":")
        print(normalized_distance_threshold)
        #Perform Clustering
        dist_condensed = ssd.squareform(DistanceMatrix_For_Clustering)
        Z = hierarchy.linkage(dist_condensed, method="average")
        attractors_clusters = hierarchy.fcluster(Z, normalized_distance_threshold, criterion='distance')
        print("Number of clusters found for " + dataset_pathway + " is : " + str(len(set(attractors_clusters))))    


        #dendograms
        if save_dendrogram:
            fig = plt.figure(figsize=(25, 10))
            dn = hierarchy.dendrogram(Z,truncate_mode=None, color_threshold=normalized_distance_threshold)
            plt.xlabel("Attractors")
            plt.ylabel("Hamming Distance")
            plt.title(dataset_pathway)
            plt.savefig(dendrogram_saving_pathway + dataset_pathway + "_merged_attractors_no_truncation_dendrogram_gk.pdf")
            plt.clf()

        if save_truncated_dendrogram:
            if isinstance(lastp_for_truncate, int):
                lastp_for_truncate = lastp_for_truncate
            else:
                lastp_for_truncate = len(set(attractors_clusters))*int(lastp_for_truncate[15:])
            fig = plt.figure(figsize=(25, 10))
            dn = hierarchy.dendrogram(Z, p=lastp_for_truncate, truncate_mode="lastp", color_threshold=normalized_distance_threshold, distance_sort=True, no_labels=False, show_contracted=True,ax=None)
            plt.xlabel("Attractors")
            plt.ylabel("Hamming Distance")
            plt.title(dataset_pathway)
            plt.savefig(dendrogram_saving_pathway + dataset_pathway + "_merged_attractors_truncation_dendrogram_gk.pdf")
            plt.clf()

    
        #group attractors within same clsuter
        map_indices_attractors_within_clusters = {} #stores dict with keys being clusters ids and values being lists which include int formated attractors within that cluster for current dataset_network pair
        for each_attr_indx in range(len(attractors_clusters)):
            if attractors_clusters[each_attr_indx] in map_indices_attractors_within_clusters:
                map_indices_attractors_within_clusters[attractors_clusters[each_attr_indx]].append(all_attractors_mapped_to_by_cells[dataset_pathway][each_attr_indx])
            else:
                map_indices_attractors_within_clusters[attractors_clusters[each_attr_indx]] = []
                map_indices_attractors_within_clusters[attractors_clusters[each_attr_indx]].append(all_attractors_mapped_to_by_cells[dataset_pathway][each_attr_indx])

        #store map_indices_attractors_within_clusters for each dataset_network pair
        set_of_clusters_per_dataset[dataset_pathway] = map_indices_attractors_within_clusters


    return set_of_clusters_per_dataset

##########################
##### Third function #####
##########################

def Merge_Similar_Attractors(set_of_clusters_per_dataset, loaded_dist_dfs,attractors_per_dtst_ntrk_saving_path):

    #merge attractors within same cluster
    set_of_clusters_with_representative_attractors_per_dataset = {}
    for dataset_pathway in set_of_clusters_per_dataset: #go through each dataset__network
        #obtain distance matrix for each attractor and its distance from each cell
        distances_df = loaded_dist_dfs[dataset_pathway] #get distances csv for current dataset__network
        cells_list = distances_df.index.tolist() #get list of cells for the current dataset
        print("Done obtaining distances matrix for " + dataset_pathway)

        representative_attractors_for_each_cluster = {}
        cluster_counter = 1
        for each_cluster in set_of_clusters_per_dataset[dataset_pathway]: #go through each cluster id in current dataset__network
            attractors_in_this_cluster = set_of_clusters_per_dataset[dataset_pathway][each_cluster]
            #First calculate distance between attractors within this cluster and cells
            cell_counter_for_attractors = [0] * len(attractors_in_this_cluster) #counts number of cells mapping to each attractor within current cluster
            for each_cell in cells_list:
                distance_of_this_cell_from_each_attractor = [0] * len(attractors_in_this_cluster)
                for each_attrac_indx in range(len(attractors_in_this_cluster)):
                    distance_attrac_cell = distances_df.loc[each_cell,str(attractors_in_this_cluster[each_attrac_indx])]
                    distance_of_this_cell_from_each_attractor[each_attrac_indx] = distance_attrac_cell
                min_index = distance_of_this_cell_from_each_attractor.index(min(distance_of_this_cell_from_each_attractor)) #returns first occurance of minimum value
                cell_counter_for_attractors[min_index] += 1
            chosen_attrac_indx = cell_counter_for_attractors.index(max(cell_counter_for_attractors))
            representative_attractor = attractors_in_this_cluster[chosen_attrac_indx]
            #store cluster id and corresponding int formatted representative attractor
            representative_attractors_for_each_cluster[each_cluster] = representative_attractor
            print("Calculated Representative Attractor for Cluster #" + str(cluster_counter) + " Out Of " + str(len(set_of_clusters_per_dataset[dataset_pathway])) + " Clusters", end = "\r")
            cluster_counter += 1
        print("Calculated Representative Attractor for Cluster #" + str(len(set_of_clusters_per_dataset[dataset_pathway])) + " Out Of " + str(len(set_of_clusters_per_dataset[dataset_pathway])) + " Clusters", end = "\r")


        set_of_clusters_with_representative_attractors_per_dataset[dataset_pathway] = representative_attractors_for_each_cluster #stores dataset__network as keys and values being a dictionary that has keys which are clusters' ids and values which are int formatted representative attractor
        print("Done Finding Representative Attractors in Each Cluster")
    
    #stores a csv with columns as cluster ids and rows being 0/1 values of representative attractors
    for dtst_ntrk in set_of_clusters_with_representative_attractors_per_dataset:
        df_dtst_ntrk = pd.DataFrame.from_dict(set_of_clusters_with_representative_attractors_per_dataset[dtst_ntrk])
        df_dtst_ntrk.to_csv(attractors_per_dtst_ntrk_saving_path+dtst_ntrk+".csv",index=False)
    
    return set_of_clusters_with_representative_attractors_per_dataset

###########################
##### Fourth function #####
###########################

def Create_Bar_Plot_And_Frequency_Plot(set_of_clusters_with_representative_attractors_per_dataset, loaded_dist_dfs, plots_saving_path, attractors_representing_clusters_indices_to_plot, datasets_cells_labels, restrict_attractors_plotted=False, Combined_Plot=True, bar_box_plots_fig_size=(3,2)):

    for dataset_pathway in set_of_clusters_with_representative_attractors_per_dataset: #go through each dataset_network pair
        representative_attractors_for_each_cluster = set_of_clusters_with_representative_attractors_per_dataset[dataset_pathway] #stores dict of cluster ids and values are representative attractors
        dist_pd = loaded_dist_dfs[dataset_pathway] #get distances csv for current dataset__network
        cells_list = dist_pd.index.tolist() #get list of cells for the current dataset
        

        Representative_Attractors = [] #representative attractors
        Representative_Attractors_Clusters_Labels = [] #the cluster id which has the representative attractor
        for representative_attrac in representative_attractors_for_each_cluster:
            Representative_Attractors.append(representative_attractors_for_each_cluster[representative_attrac])
            Representative_Attractors_Clusters_Labels.append(representative_attrac)
        print(Representative_Attractors_Clusters_Labels)

        num_nodes_in_ntrk = len(Representative_Attractors[0]) #stores number of nodes in this network
        print("num_nodes_in_ntrk: " + str(num_nodes_in_ntrk))
        
        
        dtst_name = dataset_pathway[0:(dataset_pathway.find("__"))] #stores name of dataset
        #Store attractors and their clusters for current network
        ntrk_attractors_list = []
        ntrk_attractors_labels = []
        for cluster_id in set_of_clusters_with_representative_attractors_per_dataset[dataset_pathway]:
            ntrk_attractors_labels.append(cluster_id)
            ntrk_attractors_list.append(set_of_clusters_with_representative_attractors_per_dataset[dataset_pathway][cluster_id])
        print(ntrk_attractors_labels)
        #get labels of cell types for the datasett
        label1_txt = datasets_cells_labels[dtst_name]["Label1"]
        label2_txt = datasets_cells_labels[dtst_name]["Label2"]
        #calculate distances between each type of cells and attractors
        label1_cells_counter_for_attractors = [0] * len(ntrk_attractors_list)
        label2_cells_counter_for_attractors = [0] * len(ntrk_attractors_list)
        label1_cells_distances_from_attractors = {}
        label2_cells_distances_from_attractors = {}
        #initialize distances dictionaries
        for attr_lbl in ntrk_attractors_labels:
            label1_cells_distances_from_attractors[attr_lbl] = []
            label2_cells_distances_from_attractors[attr_lbl] = []

        #to store cells' names and distances for each attractors
        label1_cells_list_for_attractors = {}
        label2_cells_list_for_attractors = {}
        for attr_id in ntrk_attractors_labels:
            label1_cells_list_for_attractors[attr_id] = []
            label2_cells_list_for_attractors[attr_id] = []
            
        for each_cell in cells_list:
            distance_of_this_cell_from_each_attractor = [0] * len(ntrk_attractors_list) 
            for each_attrac_indx in range(len(ntrk_attractors_list)):
                distance_attrac_cell = dist_pd.loc[each_cell,str(ntrk_attractors_list[each_attrac_indx])]
                distance_of_this_cell_from_each_attractor[each_attrac_indx] = distance_attrac_cell
            min_dist_value = min(distance_of_this_cell_from_each_attractor)
            min_index = distance_of_this_cell_from_each_attractor.index(min(distance_of_this_cell_from_each_attractor)) #returns first occurance of minimum value
            if label1_txt in each_cell:
                label1_cells_counter_for_attractors[min_index] += 1
                if ntrk_attractors_labels[min_index] not in label1_cells_distances_from_attractors:
                    label1_cells_distances_from_attractors[ntrk_attractors_labels[min_index]] = []
                label1_cells_distances_from_attractors[ntrk_attractors_labels[min_index]].append(min_dist_value)
                label1_cells_list_for_attractors[ntrk_attractors_labels[min_index]].append([each_cell,min_dist_value])#store cell names and DISTANCES BETWEEN CELL AND ATTRACTOR
            elif label2_txt in each_cell:
                label2_cells_counter_for_attractors[min_index] += 1
                if ntrk_attractors_labels[min_index] not in label2_cells_distances_from_attractors:
                    label2_cells_distances_from_attractors[ntrk_attractors_labels[min_index]] = []
                label2_cells_distances_from_attractors[ntrk_attractors_labels[min_index]].append(min_dist_value)
                label2_cells_list_for_attractors[ntrk_attractors_labels[min_index]].append([each_cell,min_dist_value])#store cell names and DISTANCES BETWEEN CELL AND ATTRACTOR
            else:
                print("Could Not Identify Identity Of Cell!")
        #Calculate percentages from counts
        attracs_list = tuple(ntrk_attractors_list)
        label1_cells_counter = tuple(label1_cells_counter_for_attractors)
        label2_cells_counter = tuple(label2_cells_counter_for_attractors)
        sum_label1_cells = sum(label1_cells_counter)
        sum_label2_cells = sum(label2_cells_counter)
        print("label1: " + str(sum_label1_cells))
        print("label2: " + str(sum_label2_cells))
        label1_cells_percentages = [(i1/sum_label1_cells)*100 for i1 in label1_cells_counter]
        label2_cells_percentages = [(i2/sum_label2_cells)*100 for i2 in label2_cells_counter]  
        print(label1_cells_percentages)
        print(label2_cells_percentages)
        for i in label1_cells_distances_from_attractors:
            if len(label1_cells_distances_from_attractors[i]) < 1:
                print("No Cells Mapped to Attractor " + str(i))
                print("Adding a nan value to this attractor's distances for it to still be plotted")
                label1_cells_distances_from_attractors[i].append(np.nan)
                print(label1_cells_distances_from_attractors[i])
        print("\n\n")
        for i in label2_cells_distances_from_attractors:
            if len(label2_cells_distances_from_attractors[i]) < 1:
                print("No Cells Mapped to Attractor " + str(i))
                print("Adding a nan value to this attractor's distances for it to still be plotted")
                label2_cells_distances_from_attractors[i].append(np.nan)
                print(label2_cells_distances_from_attractors[i])
        
        
        #pickle cells' names for each attracctor for this dataset_pathway as dictionaries        
        with open("Cells_Names_Per_Attractors/"+ "Label1_" + label1_txt + "_Dataset_Pathway_" + dataset_pathway + '_cells_name_in_attractors.pickle', 'wb') as handle1:
            pickle.dump(label1_cells_list_for_attractors, handle1, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open("Cells_Names_Per_Attractors/"+ "Label2_" + label2_txt + "_Dataset_Pathway_" + dataset_pathway + '_cells_name_in_attractors.pickle', 'wb') as handle2:
            pickle.dump(label2_cells_list_for_attractors, handle2, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        with open("Cells_Names_Per_Attractors/" + "num_nodes_in_ntrk_Dataset_Pathway_" + dataset_pathway + '.pickle', 'wb') as handle3:
            pickle.dump(num_nodes_in_ntrk, handle3, protocol=pickle.HIGHEST_PROTOCOL)

        #chi-sqaured test
        obs = np.array([list(label1_cells_counter), list(label2_cells_counter)])
        print(label1_cells_counter, label2_cells_counter)
        g, p, dof, expctd = chi2_contingency(obs)
        print("python test:")
        print(label1_cells_counter)
        print(label2_cells_counter)
        print(p)
        print("{0:.15f}".format(p))
        print(g)
        print(dof)
        print(expctd)

        

        
        if Combined_Plot == True:
            #create box plots and frequency plots
            #for boxplots
            labels_list_for_all_repr_attr = []
            distances_list_for_all_repr_attr = []
            cell_type_list_for_all_repr_attr = []
            ##### LABEL1 #####
            for each_repr_attr in label1_cells_distances_from_attractors:
                labels_list_for_repr_attr = [each_repr_attr]*len(label1_cells_distances_from_attractors[each_repr_attr])
                labels_list_for_all_repr_attr = labels_list_for_all_repr_attr + labels_list_for_repr_attr
                cell_type_list_for_repr_attr = [label1_txt]*len(label1_cells_distances_from_attractors[each_repr_attr])
                cell_type_list_for_all_repr_attr = cell_type_list_for_all_repr_attr + cell_type_list_for_repr_attr
                distances_list_for_all_repr_attr = distances_list_for_all_repr_attr + label1_cells_distances_from_attractors[each_repr_attr]
            ##### LABEL2 #####
            for each_repr_attr_L2 in label2_cells_distances_from_attractors:
                labels_list_for_repr_attr_L2 = [each_repr_attr_L2]*len(label2_cells_distances_from_attractors[each_repr_attr_L2])
                labels_list_for_all_repr_attr = labels_list_for_all_repr_attr + labels_list_for_repr_attr_L2
                cell_type_list_for_repr_attr_L2 = [label2_txt]*len(label2_cells_distances_from_attractors[each_repr_attr_L2])
                cell_type_list_for_all_repr_attr = cell_type_list_for_all_repr_attr + cell_type_list_for_repr_attr_L2
                distances_list_for_all_repr_attr = distances_list_for_all_repr_attr + label2_cells_distances_from_attractors[each_repr_attr_L2]
            ##### BOTH LABELS #####    
            distances_list_for_all_repr_attr = list((np.array(distances_list_for_all_repr_attr)/num_nodes_in_ntrk)*100)
            df_for_plotting = pd.DataFrame.from_dict({"representative_attractor_label":labels_list_for_all_repr_attr, "distances_from_mapped_cells":distances_list_for_all_repr_attr,"cell_type_label":cell_type_list_for_all_repr_attr})
            sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":8})

    
            #for frequency plots
            ##### LABEL1 #####
            Y5_L1 = [round(i,1) for i in label1_cells_percentages]
            X5_L1 = np.array(ntrk_attractors_labels)-0.2-1 #-1 just to fit on the axis created by sns for the boxplot. For the correct attactors' labels see other axis and not this one.
            ##### LABEL2 #####
            Y5_L2 = [round(i,1) for i in label2_cells_percentages]
            X5_L2 = np.array(ntrk_attractors_labels)+0.2-1 #-1 just to fit on the axis created by sns for the boxplot. For the correct attactors' labels see other axis and not this one.
            

            #combined
            fig = plt.figure(figsize=(bar_box_plots_fig_size))
            grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
            ax_main = fig.add_subplot(grid[1:4, 0:4])
            ax_top = fig.add_subplot(grid[0:1,0:4])
            sns.boxplot(x='representative_attractor_label',y='distances_from_mapped_cells', data=df_for_plotting, ax=ax_main, hue="cell_type_label",showfliers=False, palette={label1_txt:"#cecfcf",label2_txt:"#252525"}) #palette=["tab:blue"]
            #sns.stripplot(x="representative_attractor_label", y="distances_from_mapped_cells",data=df_for_plotting, ax=ax_main, jitter=True, marker='o', alpha=0.5, color='black')
            #color was tab:blue and tab:orange
            bar_container_L1 = ax_top.bar(X5_L1, Y5_L1, 0.4, color = '#cecfcf', label = label1_txt,edgecolor = "black")
            bar_container_L2 = ax_top.bar(X5_L2, Y5_L2, 0.4, color = '#252525', label = label2_txt,edgecolor = "black")
            ax_top.get_shared_x_axes().join(ax_top, ax_main)
            ax_top.set(title=dataset_pathway, xlabel='', ylabel='Percentage\nof Cells')
            ax_top.set_ylabel('Percentage\nof Cells', rotation=90, fontsize=8, labelpad=5)
            ax_top.set_title(dataset_pathway, fontsize=8, pad=5)
            #ax_top.set_xticks(list(range(1,len(X5)+1)))
            ax_top.set_xticks([])
            #ax_top.set_xticklabels(list(range(1,len(X5)+1)))
            plt.setp(ax_top.get_xticklabels(), visible=False)
            #ax_top.grid(color='gray',axis='y',alpha=0.3)
            ax_top.bar_label(bar_container_L1, label_type='edge', size=7)
            ax_top.bar_label(bar_container_L2, label_type='edge', size=7)
            max_y_top = int(max(Y5_L1+Y5_L2))
            min_y_top = int(min(Y5_L1+Y5_L2))
            step_y_top = 20
            ax_top.set_yticks(list(range(0,121,step_y_top)))
            ax_top.set_yticklabels([str(i) for i in list(range(0,101,step_y_top))]+[""], fontsize=7)
            ax_main.set(xlabel='Representative Attractors', ylabel='Normalized Distances\nFrom Mapped Cells')
            ax_main.set_ylabel('Normalized Distances\nFrom Mapped Cells', rotation=90, fontsize=8, labelpad=5)
            ax_main.set_xlabel('Representative Attractors', fontsize=8, labelpad=5)
            max_y = int(max(list(df_for_plotting["distances_from_mapped_cells"])))
            min_y = int(min(list(df_for_plotting["distances_from_mapped_cells"])))
            if (max_y-min_y) < 20:
                step_y = 2
            else:
                step_y = 5
            ax_main.set_yticks(list(range(min_y,max_y+1,step_y)))
            ax_main.set_yticklabels(list(range(min_y,max_y+1,step_y)))
            ax_main.grid(color='gray',axis='y',alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_saving_path + dataset_pathway + "_combined_plot.pdf")
            plt.clf()


            
############################################
##### Function that generated heatmaps #####
############################################

def Create_Heatmap_Across_Datasets(attractors_for_heatmap_list,labels_for_heatmap_list,dataset_addresses_for_heatmap_list,network_id_for_heatmap_list,saving_name_for_heatmap,saving_path_for_heatmap,dendrogram_or_not=False,fig_s = (2.5,6)):
    
    
    all_dfs_for_hm = []
    
    for dtst_hm_adrs_idx in range(len(dataset_addresses_for_heatmap_list)):
        objectFile = glob.glob(dataset_addresses_for_heatmap_list[dtst_hm_adrs_idx] + "*.csvscTest.pickle")
        scObject = pickle.load(open(objectFile[0], "rb"))
        #print(scObject.sampleList)
        network_graphml = network_id_for_heatmap_list[dtst_hm_adrs_idx] + "_processed.graphml"
        net = nx.read_graphml(dataset_addresses_for_heatmap_list[dtst_hm_adrs_idx]+network_graphml)
        #Create ruleMaker object
        scObject._singleCell__inherit(net, removeSelfEdges=False, restrictIncomingEdges=True, maxIncomingEdges=3, groundTruth=False)
        scObject._ruleMaker__updateCpointers()
        scObject.knockoutLists = [0] * len(scObject.nodePositions)
        scObject.knockinLists = [0] * len(scObject.nodePositions)
        #print(net)
        #heatmaps for specified attractors
        number_attractors = len(attractors_for_heatmap_list[dtst_hm_adrs_idx])
        attractorList = attractors_for_heatmap_list[dtst_hm_adrs_idx]
        #print(number_attractors)
        #print(attractorList)
        attractorDF = pd.DataFrame(attractorList).T
        #print(attractorDF.shape)
        index_sc = [scObject.geneList[temp] for temp in scObject.nodePositions]
        attractorDF.index = index_sc
        attractorDF.columns = labels_for_heatmap_list[dtst_hm_adrs_idx]
        all_dfs_for_hm.append(attractorDF)
        
    
    #merge hm dfs across datasets
    global_df_hm = all_dfs_for_hm[0]
    if len(all_dfs_for_hm) > 1:
        print("Working on more than one dataset!")
        for df_hm in all_dfs_for_hm[1:len(all_dfs_for_hm)]:
            global_df_hm = global_df_hm.merge(df_hm,how="outer",left_index=True,right_index=True)
            
    
    #remove genes/rows same across attractors
    attractorDF_chosen = global_df_hm[global_df_hm.apply(lambda x: min(x) != max(x), 1)]
    
    #print(global_df_hm)
    
    sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":8})
    cmap = ListedColormap(sns.color_palette(["#E1F5FE", "#0277BD"]))

    #create mask
    g_df_shape = attractorDF_chosen.shape
    mask = pd.DataFrame(np.zeros(g_df_shape,dtype=int))
    for r_i in range(g_df_shape[0]):
        for c_i in range(g_df_shape[1]):
            if ((attractorDF_chosen.iloc[r_i,c_i] == 0.0) or (attractorDF_chosen.iloc[r_i,c_i] == 1.0)):
                mask.iloc[r_i,c_i] = True
            else:
                attractorDF_chosen.iloc[r_i,c_i] = np.nan
           

    #group similar genes together
    new_df_lt = []
    new_genes_lt = []
    df_genes_lst = list(attractorDF_chosen.index)
    df_attractors_lst = list(attractorDF_chosen.columns)
    for gene_i in df_genes_lst:
        if gene_i == "USED":
            continue
        new_genes_lt.append(gene_i)
        
        new_df_lt_row_holder = list(attractorDF_chosen.loc[gene_i])
        new_df_lt_row = []
        for gn_val_idx in range(len(new_df_lt_row_holder)):
            if new_df_lt_row_holder[gn_val_idx] == 0.0:
                new_df_lt_row.append(0.0)
            elif new_df_lt_row_holder[gn_val_idx] == 1.0:
                new_df_lt_row.append(1.0)
            else:
                new_df_lt_row.append(np.nan)
        new_df_lt.append(new_df_lt_row)
        
        df_genes_lst[df_genes_lst.index(gene_i)] = "USED"
        similar_genes_string = ""
        for gene_j in df_genes_lst:
            if gene_j=="USED":
                continue
            else:
                #prepare lists for comparison although they includes nan values
                alt1_pre = list(attractorDF_chosen.loc[gene_i])
                alt2_pre = list(attractorDF_chosen.loc[gene_j])
                alt1 = []
                alt2 = []
                #comparison 1 list
                for i in range(len(alt1_pre)):
                    if alt1_pre[i] == 0.0:
                        alt1.append(0.0)
                    elif alt1_pre[i] == 1.0:
                        alt1.append(1.0)
                    else:
                        alt1.append(np.nan)
                #comparison 2 list
                for i in range(len(alt2_pre)):
                    if alt2_pre[i] == 0.0:
                        alt2.append(0.0)
                    elif alt2_pre[i] == 1.0:
                        alt2.append(1.0)
                    else:
                        alt2.append(np.nan)
                if (alt1 == alt2):
                    df_genes_lst[df_genes_lst.index(gene_j)] = "USED"
                    similar_genes_string = similar_genes_string + ", " + gene_j
        new_genes_lt[new_genes_lt.index(gene_i)] = new_genes_lt[new_genes_lt.index(gene_i)] + similar_genes_string
    #add new lines in new indices
    for new_gn in new_genes_lt:
        fixed_new_gn = ""
        counter_gns = 0
        for letter_idx in range(len(new_gn)):
            if new_gn[letter_idx] == ",":
                counter_gns += 1
            if counter_gns==3:
                counter_gns = 0
                fixed_new_gn = fixed_new_gn + ",\n"
            else:
                fixed_new_gn = fixed_new_gn + new_gn[letter_idx]
        new_genes_lt[new_genes_lt.index(new_gn)] = fixed_new_gn
        
        
    #group names more than 9 genes
    for new_gn in new_genes_lt:
        if (new_gn.count(',') > 5):
            genes_group_name = input("Input a group name for the genes: " + new_gn)
            print(genes_group_name + " : " + new_gn)
            new_genes_lt[new_genes_lt.index(new_gn)] = genes_group_name
    
            
    #new dataframe
    new_df_holder = pd.DataFrame(new_df_lt, columns = df_attractors_lst)
    new_df_holder.index = new_genes_lt       
    attractorDF_chosen = new_df_holder
    print(new_genes_lt)

  
    ax = sns.clustermap(attractorDF_chosen, method="average", cmap=cmap, row_cluster=False, col_cluster=False, dendrogram_ratio=0.2, linewidth=0.3, yticklabels=True, xticklabels=True, vmin = 0, vmax=1, cbar_kws={"ticks":[0,1]}, figsize=fig_s,cbar_pos=None) #cbar_pos=None to disable it #mask=mask
    #ax.cax.set_position([.15, .2, .03, .45])
    #ax.cax.yaxis.set_label_position("left")
    fig_hm = ax.fig
    fig_hm.suptitle(saving_name_for_heatmap, fontsize=8)
    ax.ax_col_dendrogram.set_visible(dendrogram_or_not)
    #ax.ax_row_dendrogram.set_visible(dendrogram_or_not)
    current_fig = ax.figure
    plt.tight_layout()
    current_fig.savefig(saving_path_for_heatmap + saving_name_for_heatmap + "_attractors_across_datasets_heatmap.pdf")
    plt.clf()
        

        
        
        
        
###########################################################################################
##### Function that generated bar and box plots mapping attractors to B cell subtypes #####
###########################################################################################

def Check_Percentage_of_Certain_Cells_in_Attractors(azimuth_cells_csv,exprs_df_address,dataset_pathway_address_L1,dataset_pathway_address_L2,num_nodes_in_ntrk_address,cells_criteria_list,attractors_list,color_map_for_labels,criteria_cells_saving_path,dataset_pathway_label,dtst_name,cell_type = "all", combined_plot = True, outside_reference = True, cell_types_cells_names = {},subtypes_fig_size=(6,4)):
    

    if outside_reference==False:
        #import attractors and cells expression csv
        print("Importing Expression CSV File")
        genes_cells_exprs_df = pd.read_csv(exprs_df_address,index_col=0)  
        genes_names_df = list(genes_cells_exprs_df.index.values)
        all_cells_df_list = list(genes_cells_exprs_df.columns)
        all_cells_sum = len(all_cells_df_list)
        print("Done Importing Expression CSV File")
    else:
        print("Importing cells' names")
        az_df = pd.read_csv(azimuth_cells_csv,index_col=0)
        all_cells_df_list = list(az_df.index)
        all_cells_sum = len(all_cells_df_list)
    
    
    #import pickle of attractors and cells' names mapping to them across both types of cells
    with open(dataset_pathway_address_L1, 'rb') as handle1:
        prefilter_dict_L1 = pickle.load(handle1)
    postfilter_L1 = {}
    for attr_ID in attractors_list:
        postfilter_L1[attr_ID] = prefilter_dict_L1[attr_ID]
        
    with open(dataset_pathway_address_L2, 'rb') as handle2:
        prefilter_dict_L2 = pickle.load(handle2)
    postfilter_L2 = {}
    for attr_ID in attractors_list:
        postfilter_L2[attr_ID] = prefilter_dict_L2[attr_ID]
    
    attractors_cells_dict_L1 = postfilter_L1
    attractors_cells_dict_L2 = postfilter_L2
    
    with open(num_nodes_in_ntrk_address, 'rb') as handle3:
        num_nodes_in_nrk = pickle.load(handle3)

    combined_attractors_cells_dict = {}
    for attr_id in attractors_cells_dict_L1:
        combined_attractors_cells_dict[attr_id] = attractors_cells_dict_L1[attr_id] + attractors_cells_dict_L2[attr_id]
        
    #Specifies if wants to work with only a specific type of cells (e.g. diseased, healthy, or both)
    if cell_type == "all":
        final_attractors_cells_dict = combined_attractors_cells_dict
    elif cell_type == "label1":
        final_attractors_cells_dict = attractors_cells_dict_L1
    elif cell_type == "label2":
        final_attractors_cells_dict = attractors_cells_dict_L2
        
    #attractors_cells_counts_dict = {1:{"label1":30,"label2":55},2:{"label1":84,"label2":10}} 
    # attractors_cells_dist_dict = {1:{"label1":[1,2,3],"label2":[4,5,6]},2:{"label1":[1,2,3],"label2":[4,5,6]}} 
    #Calculate pecentages and distances
    attractors_cells_counts_dict = {}
    for at_id in final_attractors_cells_dict:
        attractors_cells_counts_dict[at_id] = {}
        for c_tp in cells_criteria_list:
            attractors_cells_counts_dict[at_id][c_tp] = 0
            
    counts_for_chi_squared_test = {}
    for at_id in final_attractors_cells_dict:
        counts_for_chi_squared_test[at_id] = {}
        for c_tp in cells_criteria_list:
            counts_for_chi_squared_test[at_id][c_tp] = 0
            
    if combined_plot == True:
        attractors_cells_dist_dict = {}
        for at_id in final_attractors_cells_dict:
            attractors_cells_dist_dict[at_id] = {}
            for c_tp in cells_criteria_list:
                attractors_cells_dist_dict[at_id][c_tp] = []
                             
    all_cell_types_names = []
    for i in cell_types_cells_names:
        all_cell_types_names = all_cell_types_names + cell_types_cells_names[i]

    #loop through all attractors
    for atrctr_id in final_attractors_cells_dict:
        print("Calculating for attractor " + str(atrctr_id))
        #store a list of cells' names that mapped this attractor
        gt1 = final_attractors_cells_dict[atrctr_id]
        cells_for_this_atrctr_list = [i[0] for i in gt1]
        #print("cells_for_this_atrctr_list:")
        #print(cells_for_this_atrctr_list)
        cells_dists_for_this_atrctr_list = [i[1] for i in gt1]
        num_cells_for_this_atrctr = len(cells_for_this_atrctr_list)
        #print(num_cells_for_this_atrctr)
        #print(cells_for_this_atrctr_list)
        #cells that had a type
        typed_cells_for_this_atr = []
        #loop through all cells with different criteria
        for cell_type in cells_criteria_list:
            print("Calculating for cell type " + str(cell_type))
            #store this cell type criteria
            criteria_for_this_cell_type = cells_criteria_list[cell_type]
            #loop through each cell in this attractor and test if it meets criteria
            for each_cell_name_idx in range(len(cells_for_this_atrctr_list)):
                if outside_reference == False:
                    genes_on_criteria_met = 1
                    genes_off_criteria_met = 1
                    #test genes on
                    if len(criteria_for_this_cell_type["on"]) > 0:
                        on_genes = criteria_for_this_cell_type["on"]
                        for on_gn in on_genes:
                            if on_gn not in genes_names_df:
                                if each_cell_name_idx == 0:
                                    print("Gene " + on_gn + " NOT Found")
                                genes_on_criteria_met = 0
                                continue
                            if genes_cells_exprs_df.loc[on_gn,cells_for_this_atrctr_list[each_cell_name_idx]] > 0.001:
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
                            if off_gn not in genes_names_df:
                                if each_cell_name_idx == 0:
                                    print("Gene " + off_gn + " NOT Found")
                                genes_off_criteria_met = 0
                                continue
                            if genes_cells_exprs_df.loc[off_gn,cells_for_this_atrctr_list[each_cell_name_idx]] <= 0.001:
                                continue
                            else:
                                genes_off_criteria_met = 0
                                break
                    #if both criteria are met then this cell's info should be added to percs and dist dictionaries
                    if (genes_on_criteria_met == 1) and (genes_off_criteria_met == 1):
                        attractors_cells_counts_dict[atrctr_id][cell_type] += 1
                        counts_for_chi_squared_test[atrctr_id][cell_type] += 1
                        typed_cells_for_this_atr.append(cells_for_this_atrctr_list[each_cell_name_idx])
                        if combined_plot == True:
                            cell_attr_dist = cells_dists_for_this_atrctr_list[each_cell_name_idx]
                            attractors_cells_dist_dict[atrctr_id][cell_type].append(cell_attr_dist)

                    print("Tested Critetira for Cell #" + str(each_cell_name_idx+1) + " Out Of " + str(len(cells_for_this_atrctr_list)) + " Cells", end = "\r")
                    
                elif outside_reference==True:
                    if cells_for_this_atrctr_list[each_cell_name_idx] in cell_types_cells_names[cell_type]:
                        attractors_cells_counts_dict[atrctr_id][cell_type] += 1
                        counts_for_chi_squared_test[atrctr_id][cell_type] += 1
                        typed_cells_for_this_atr.append(cells_for_this_atrctr_list[each_cell_name_idx])
                        if combined_plot == True:
                            cell_attr_dist = cells_dists_for_this_atrctr_list[each_cell_name_idx]
                            attractors_cells_dist_dict[atrctr_id][cell_type].append(cell_attr_dist)
                    elif cells_for_this_atrctr_list[each_cell_name_idx] not in all_cell_types_names:
                        print("Attractor cell not in one of the types!!!!")
                        

            #convert counts to percentages
            attractors_cells_counts_dict[atrctr_id][cell_type] = ((attractors_cells_counts_dict[atrctr_id][cell_type])/num_cells_for_this_atrctr)*100
            print("\n")
        print("\n\n")

    #counts_for_chi_squared_test = {1:{"label1":30,"label2":55},2:{"label1":84,"label2":10}} 
    #Chi-Square Test of Independence
    #print(counts_for_chi_squared_test)
    attractors_nums_chi_list = []
    cell_types_chi_list = []
    for attr_num_i in counts_for_chi_squared_test:
        attractors_nums_chi_list.append(attr_num_i)
    for cell_type_i in counts_for_chi_squared_test[attractors_nums_chi_list[0]]:
        cell_types_chi_list.append(cell_type_i)
        
    prepared_list_for_array = []    
    for each_atr_i in attractors_nums_chi_list:
        atr_types_list = []
        for each_cl_tp in cell_types_chi_list:
            atr_types_list.append(counts_for_chi_squared_test[each_atr_i][each_cl_tp])
        prepared_list_for_array.append(atr_types_list)
        
        
    obs = np.array(prepared_list_for_array)
    g, p, dof, expctd = chi2_contingency(obs)
    print("python test:")
    print(p)
    print("{0:.15f}".format(p))
    print(g)
    print(dof)
    print(expctd)

    
    if combined_plot == True:
        for a_i in attractors_cells_dist_dict:
            for t_i in attractors_cells_dist_dict[a_i]:    
                if len(attractors_cells_dist_dict[a_i][t_i]) < 1:
                    print("No Cells Mapped to Attractor " + str(a_i) + " Cell Type " + str(t_i))
                    print("Adding a nan value to this attractor's distances for it to still be plotted")
                    attractors_cells_dist_dict[a_i][t_i].append(np.nan)
                    print(attractors_cells_dist_dict[a_i][t_i])

    
    #bar plot preperations
    atrctrs_labels = []
    cell_type_labels = []
    cell_type_percs_across_attractors_lists = []
    cell_type_dists_across_attractors_lists = []
    #fill atrctrs_labels
    for attrc_id in attractors_cells_counts_dict:
        atrctrs_labels.append(attrc_id)
    #fil in order the types in cell_type_labels
    for attrc_id in attractors_cells_counts_dict:
        for ct_lbl in attractors_cells_counts_dict[attrc_id]:
            cell_type_labels.append(ct_lbl)
        break  
    #fill cell_type_percs_across_attractors_lists
    for ct_lbl in cell_type_labels:
        ordered_percs_across_atrctrs = []
        for attrc_id in atrctrs_labels:
            ordered_percs_across_atrctrs.append(round(attractors_cells_counts_dict[attrc_id][ct_lbl],1))
        cell_type_percs_across_attractors_lists.append(ordered_percs_across_atrctrs)
    #cell_type_percs_across_attractors_lists = [[forattr1,forattr2,...],[forattr1,forattr2,...],...]
    #fill cell_type_dists_across_attractors_lists    
    for ct_lbl in cell_type_labels:
        ordered_dists_across_atrctrs = []
        for attrc_id in atrctrs_labels:
            ordered_dists_across_atrctrs.append(attractors_cells_dist_dict[attrc_id][ct_lbl])
        cell_type_dists_across_attractors_lists.append(ordered_dists_across_atrctrs)
    
    
    #plotting combined
    if combined_plot == True:
        #create box plots and frequency plots
        #for boxplots
        labels_list_for_all_repr_atr = []
        cell_type_list_for_all_repr_atr = []
        distances_list_for_all_repr_atr = []
        for each_repr_atr in attractors_cells_dist_dict:
            for each_cl_typ in attractors_cells_dist_dict[each_repr_atr]:
                #if len(attractors_cells_dist_dict[each_repr_atr][each_cl_typ]) < 1:
                #    continue
                for each_cl_dist in attractors_cells_dist_dict[each_repr_atr][each_cl_typ]:
                    labels_list_for_all_repr_atr.append(each_repr_atr)
                    cell_type_list_for_all_repr_atr.append(each_cl_typ)
                    distances_list_for_all_repr_atr.append(each_cl_dist)
                    
        #adjusting attractors' labels to share axes
        list_of_x_axis_per_atr_labels = (np.arange(0,len(atrctrs_labels)*2,2)+((len(cell_type_labels)/2)*0.3))
        map_atra_id_to_x_tick_dict = {}
        for atra_idx in range(len(atrctrs_labels)):
            map_atra_id_to_x_tick_dict[atrctrs_labels[atra_idx]] = list_of_x_axis_per_atr_labels[atra_idx]
        for atr_lbl_indx in range(len(labels_list_for_all_repr_atr)):
            holder_var = labels_list_for_all_repr_atr[atr_lbl_indx]
            labels_list_for_all_repr_atr[atr_lbl_indx] = map_atra_id_to_x_tick_dict[holder_var]
        
        ##### Create Data Frame For Plotting #####   
        distances_list_for_all_repr_atr = list((np.array(distances_list_for_all_repr_atr)/num_nodes_in_nrk)*100)
        df_for_pltng = pd.DataFrame.from_dict({"representative_attractor_label":labels_list_for_all_repr_atr, "distances_from_mapped_cells":distances_list_for_all_repr_atr,"cell_type_label":cell_type_list_for_all_repr_atr})
        sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":8})

        #combined
        fig = plt.figure(figsize=subtypes_fig_size)
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
        ax_top_2 = fig.add_subplot(grid[0:1,0:4])
        ax_main_2 = fig.add_subplot(grid[1:4, 0:4])#sharex=ax_top_2
        X_axis = np.array(list(range(0,(2*len(atrctrs_labels)),2)))
        type_ctr = 0
        for ech_type_indx in range(len(cell_type_labels)):
            crnt_type_ordered_atrctrs_cells_percs = cell_type_percs_across_attractors_lists[ech_type_indx]
            bar_container_2 = ax_top_2.bar(list(X_axis+(type_ctr*0.3)), crnt_type_ordered_atrctrs_cells_percs, 0.3, color=color_map_for_labels[cell_type_labels[ech_type_indx]], label=cell_type_labels[ech_type_indx], edgecolor = "black")
            type_ctr += 1
            ax_top_2.bar_label(bar_container_2, label_type='edge', size=8)
            
        type_ctr_2 = 0
        for ech_type_indx in range(len(cell_type_labels)):
            crnt_type_ordered_atrctrs_cells_dists = cell_type_dists_across_attractors_lists[ech_type_indx]
            bplot1 = ax_main_2.boxplot(crnt_type_ordered_atrctrs_cells_dists, positions=list(X_axis+(type_ctr_2*0.3)), widths=0.3, showfliers=False, patch_artist=True)
            type_ctr_2 += 1
            # fill with colors
            colors_list = [color_map_for_labels[cell_type_labels[ech_type_indx]]] * len(atrctrs_labels)
            for patch, color in zip(bplot1['boxes'], colors_list):
                patch.set_facecolor(color)
            for median in bplot1['medians']:
                median.set_color('black')

        #share axes
        ax_top_2.get_shared_x_axes().join(ax_top_2, ax_main_2)
        ax_top_2.set(title=dtst_name, xlabel='', ylabel='Percentage\nof Cells')
        ax_top_2.set_ylabel('Percentage\nof Cells', rotation=90, fontsize=8, labelpad=5)
        ax_top_2.set_title(dtst_name, fontsize=8, pad=5)
        #ax_top.set_xticks(list(range(1,len(X5)+1)))
        
        #x axis for top plot
        ax_top_2.set_xticks([])
        plt.setp(ax_top_2.get_xticklabels(), visible=False)
        
        #x axis for main plot
        ax_main_2.set_xticks((np.arange(0,len(atrctrs_labels)*2,2)+((len(cell_type_labels)/2)*0.3)))
        ax_main_2.set_xticklabels(atrctrs_labels)
        
        #ax_top.grid(color='gray',axis='y',alpha=0.3)
        step_y_top = 20
        ax_top_2.set_yticks(list(range(0,121,step_y_top)))
        ax_top_2.set_yticklabels([str(i) for i in list(range(0,101,step_y_top))]+[""], fontsize=7)
        ax_main_2.set(xlabel='Representative Attractors', ylabel='Normalized Distances\nFrom Mapped Cells')
        ax_main_2.set_ylabel('Normalized Distances\nFrom Mapped Cells', rotation=90, fontsize=8, labelpad=5)
        ax_main_2.set_xlabel('Representative Attractors', fontsize=8, labelpad=5)
        step_y = 5
        ax_main_2.grid(color='gray',axis='y',alpha=0.3)
        plt.tight_layout()
        plt.savefig(criteria_cells_saving_path + dtst_name + "_combined_plot.pdf")
        plt.clf()
        
#####################################################################
##### Function that plots Azimuth prediction and mapping scores #####
#####################################################################

def plot_azimuth_scores(datasets_prediction_scores_addresses,datasets_mapping_scores_addresses,datasets_labels,datasets_color_map, plot_type="violin",figure_dims=(10,10), scale_violin="width"):
    #preparing data
    prediction_scores_for_plotting = []
    mapping_scores_for_plotting = []
    for dtst_indx in range(len(datasets_labels)):
        prediction_scores_dtst_df = pd.read_csv(datasets_prediction_scores_addresses[dtst_indx],index_col=0)
        mapping_scores_dtst_df = pd.read_csv(datasets_mapping_scores_addresses[dtst_indx],index_col=0)
        prediction_scores_dtst_data = list(prediction_scores_dtst_df["prediction_scores"])
        mapping_scores_dtst_data = list(mapping_scores_dtst_df["mapping_scores"])
        prediction_scores_for_plotting.append(prediction_scores_dtst_data)
        mapping_scores_for_plotting.append(mapping_scores_dtst_data)

    if plot_type=="violin":
        ##### VIOLIN PLOT #####
        fig = plt.figure(figsize=figure_dims)
        data_dict_for_df = {"Azimuth Scores":[],"Scores Type":[],"Datasets":[]}
        for data_indx in range(len(datasets_labels)):
            dataset_label = datasets_labels[data_indx]
            assert len(prediction_scores_for_plotting[data_indx]) == len(mapping_scores_for_plotting[data_indx])
            for score_idx in range(len(prediction_scores_for_plotting[data_indx])):
                data_dict_for_df["Azimuth Scores"].append(prediction_scores_for_plotting[data_indx][score_idx])
                data_dict_for_df["Scores Type"].append("Prediction Scores")
                data_dict_for_df["Datasets"].append(dataset_label)
            for score_idx in range(len(mapping_scores_for_plotting[data_indx])):
                data_dict_for_df["Azimuth Scores"].append(mapping_scores_for_plotting[data_indx][score_idx])
                data_dict_for_df["Scores Type"].append("Mapping Scores")
                data_dict_for_df["Datasets"].append(dataset_label)
                
        data_df = pd.DataFrame.from_dict(data_dict_for_df)
        
        
        sns.violinplot(data=data_df[data_df["Scores Type"]=="Prediction Scores"], x="Azimuth Scores", y="Datasets", scale=scale_violin,inner=None, palette="colorblind")
        plt.xlabel("Azimuth Prediction Scores", fontsize=8, labelpad=5)
        plt.savefig("New_Azimuth_Output/prediction_scores_violin_plot.pdf")
        plt.clf()
        
        fig = plt.figure(figsize=figure_dims)
        sns.violinplot(data=data_df[data_df["Scores Type"]=="Mapping Scores"], x="Azimuth Scores", y="Datasets", scale=scale_violin,inner=None, palette="colorblind")
        plt.xlabel("Azimuth Mapping Scores", fontsize=8, labelpad=5)
        plt.savefig("New_Azimuth_Output/mapping_scores_violin_plot.pdf")
        plt.clf()
        
        fig = plt.figure(figsize=figure_dims)
        sns.violinplot(data=data_df, x="Azimuth Scores", y="Datasets", hue="Scores Type" , split=True, scale=scale_violin,inner=None, palette="colorblind")
        plt.xlabel("Azimuth Prediction and Mapping Scores", fontsize=8, labelpad=5)
        plt.tight_layout()
        plt.legend([],[], frameon=False)
        plt.savefig("New_Azimuth_Output/both_scores_violin_plot.pdf")
        plt.clf()
        
print("Done loading functions")
