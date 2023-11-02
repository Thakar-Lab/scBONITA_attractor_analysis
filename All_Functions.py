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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time
from scipy.stats import chi2_contingency
from matplotlib import rcParams

seed(15)  # set seed for reproducibility
print("Done importing libraries")

########################################
##### Attractor analysis functions #####
########################################

#################################################################
##### Function that imports attractors that mapped to cells #####
#################################################################


def Import_Attractors_Mapped_To_By_Cells_And_Their_Counts(
    addresses_to_datasets_files,
    pathways_ids,
    datasets_labels,
    networks_names,
    active_gene_tresh=1,
):
    all_attractors_mapped_to_by_cells = (
        {}
    )  # store the list of int formated mapped attractors (values) per dataset and network (keys where dataset and network name separated by "__")
    counts_for_attractors_mapped_to_by_cells = (
        {}
    )  # store the dictionaries of attractors_cells_count (values) per dataset and network (keys where dataset and network name separated by "__")
    loaded_dist_dfs = (
        {}
    )  # store the loaded distances df (values) per dataset and network (keys where dataset and network name separated by "__")
    datasets_addresses = (
        {}
    )  # store the address of the imported df (values) per dataset and network (keys where dataset and network name separated by "__")

    for scBonita_dataset_idx in range(
        len(addresses_to_datasets_files)
    ):  # go through address indx of each dataset
        for pathway_idx in range(
            len(pathways_ids)
        ):  # go through each network's id indx
            attractors_cells_count = (
                {}
            )  # stores string format of each attractor (keys) and the values being the number of cells maping to each of these attractors
            # generate address of cells and attractors distances for this dataset
            distances_file_address = (
                addresses_to_datasets_files[scBonita_dataset_idx]
                + pathways_ids[pathway_idx]
                + "_processed.graphml_attractorDistance.csv"
            )

            if os.path.exists(
                distances_file_address
            ):  # check that the dataset's csv file exists
                dist_df = pd.read_csv(
                    distances_file_address, index_col=0
                )  # import distances file
                mapped_attracs_idxs = dist_df[
                    "decider"
                ].tolist()  # store the indices of the attractors mapped to by cells
                mapped_attracs = (
                    []
                )  # stores the int format of attractors mapped to by cells
                for (
                    attrac_indx
                ) in (
                    mapped_attracs_idxs
                ):  # go through each attractor indx mapped to by cells (reptition allowed)
                    attractor_based_on_indx = dist_df.columns[
                        attrac_indx
                    ]  # find the 0/1s pattern for each attractor
                    attrac_holder = list(
                        map(int, (attractor_based_on_indx.strip("[]").split(", ")))
                    )  # fix format from strings to ints
                    # make sure attractor isn't all zeros
                    if sum(attrac_holder) < active_gene_tresh:
                        continue
                    # increase the count of cells per attractor
                    if attractor_based_on_indx in attractors_cells_count:
                        attractors_cells_count[attractor_based_on_indx] += 1
                    else:
                        attractors_cells_count[attractor_based_on_indx] = 1
                # go through each mapped attractor and appned it to the list mapped_attracs
                for attract in attractors_cells_count:
                    mapped_attracs.append(attract)
                mapped_attracs = [
                    list(map(int, (k.strip("[]").split(", ")))) for k in mapped_attracs
                ]  # fix format of attractors from ints to strings

                all_attractors_mapped_to_by_cells[
                    datasets_labels[scBonita_dataset_idx]
                    + "__"
                    + networks_names[pathways_ids[pathway_idx]]
                ] = mapped_attracs  # store the list of int formated mapped attractors (values) per dataset and network (keys where dataset and network name separated by "__")
                counts_for_attractors_mapped_to_by_cells[
                    datasets_labels[scBonita_dataset_idx]
                    + "__"
                    + networks_names[pathways_ids[pathway_idx]]
                ] = attractors_cells_count  # store the dictionaries of attractors_cells_count (values) per dataset and network (keys where dataset and network name separated by "__")
                loaded_dist_dfs[
                    datasets_labels[scBonita_dataset_idx]
                    + "__"
                    + networks_names[pathways_ids[pathway_idx]]
                ] = dist_df  # store the loaded distances df (values) per dataset and network (keys where dataset and network name separated by "__")
                datasets_addresses[
                    datasets_labels[scBonita_dataset_idx]
                    + "__"
                    + networks_names[pathways_ids[pathway_idx]]
                ] = addresses_to_datasets_files[
                    scBonita_dataset_idx
                ]  # store the address of the imported df (values) per dataset and network (keys where dataset and network name separated by "__")
                print(
                    "Done Importing attractors mapped to by cells for "
                    + datasets_labels[scBonita_dataset_idx]
                    + "__"
                    + networks_names[pathways_ids[pathway_idx]]
                )
            else:
                print(
                    "Pathway "
                    + networks_names[pathways_ids[pathway_idx]]
                    + " Could Not Be Found!"
                )

    return (
        all_attractors_mapped_to_by_cells,
        counts_for_attractors_mapped_to_by_cells,
        loaded_dist_dfs,
        datasets_addresses,
    )


##################################################################
##### Function that clusters attractors that mapped to cells #####
##################################################################


def Attractors_Clustering(
    all_attractors_mapped_to_by_cells,
    distance_threshold,
    dendrogram_saving_pathway,
    lastp_for_truncate,
    save_truncated_dendrogram=False,
    save_dendrogram=True,
):
    set_of_clusters_per_dataset = (
        {}
    )  # a dict of dict. outer dict includes keys which are dataset__network. inner dict stores cluster id as keys and values being list of lists. The inner lists being the attractors within that cluster

    for (
        dataset_pathway
    ) in (
        all_attractors_mapped_to_by_cells
    ):  # go through each dataset_network pair where keys are dataset and network name separated by "__". Values are list of int formated attractors mapped to by cells
        if (
            len(all_attractors_mapped_to_by_cells[dataset_pathway]) == 1
        ):  # check if there is only one attractor in list for that dataset_network pair
            # if only one attractor per dataset_network
            set_of_clusters_per_dataset[dataset_pathway] = {1: []}
            set_of_clusters_per_dataset[dataset_pathway][1].append(
                all_attractors_mapped_to_by_cells[dataset_pathway][0]
            )
            continue
        # Calculate Distance Matrix for attractors mapped to by cells
        DistanceMatrix_For_Clustering = []
        for attr1_indx in range(
            len(all_attractors_mapped_to_by_cells[dataset_pathway])
        ):  # first go through indices of int formmated mapped attractors per dataset_network pair
            attr1_distances = (
                []
            )  # stores hamming distances between first attractor and the rest of attractors
            for attr2 in all_attractors_mapped_to_by_cells[
                dataset_pathway
            ]:  # second go through of int formmated mapped attractors per dataset_network pair
                attr1_distances.append(
                    sum(
                        [
                            abs(i1 - i2)
                            for i1, i2 in zip(
                                all_attractors_mapped_to_by_cells[dataset_pathway][
                                    attr1_indx
                                ],
                                attr2,
                            )
                        ]
                    )
                )  # Calculates Hamming Distance
            DistanceMatrix_For_Clustering.append(attr1_distances)
            print(
                "Calculated Distances For Attractor #"
                + str(attr1_indx + 1)
                + " Out Of "
                + str(len(all_attractors_mapped_to_by_cells[dataset_pathway]))
                + " Attractors",
                end="\r",
            )
        print(
            "Calculated Distances For Attractor #"
            + str(len(all_attractors_mapped_to_by_cells[dataset_pathway]))
            + " Out Of "
            + str(len(all_attractors_mapped_to_by_cells[dataset_pathway]))
            + " Attractors"
        )
        print("Done calculating Distance Matrix for " + dataset_pathway)

        # normalize the distance treshold by the number of nodes in the network
        num_nodes_in_ntrk = len(all_attractors_mapped_to_by_cells[dataset_pathway][0])
        normalized_distance_threshold = int(
            (distance_threshold / 100) * num_nodes_in_ntrk
        )
        print("Normalized distance treshold for " + dataset_pathway + ":")
        print(normalized_distance_threshold)
        # Perform Clustering
        dist_condensed = ssd.squareform(DistanceMatrix_For_Clustering)
        Z = hierarchy.linkage(dist_condensed, method="average")
        attractors_clusters = hierarchy.fcluster(
            Z, normalized_distance_threshold, criterion="distance"
        )
        print(
            "Number of clusters found for "
            + dataset_pathway
            + " is : "
            + str(len(set(attractors_clusters)))
        )

        # dendograms
        if save_dendrogram:
            fig = plt.figure(figsize=(25, 10))
            dn = hierarchy.dendrogram(
                Z, truncate_mode=None, color_threshold=normalized_distance_threshold
            )
            plt.xlabel("Attractors")
            plt.ylabel("Hamming Distance")
            plt.title(dataset_pathway)
            plt.show()
            plt.savefig(
                dendrogram_saving_pathway
                + dataset_pathway
                + "_merged_attractors_no_truncation_dendrogram_gk.pdf"
            )
            plt.clf()

        if save_truncated_dendrogram:
            if isinstance(lastp_for_truncate, int):
                lastp_for_truncate = lastp_for_truncate
            else:
                lastp_for_truncate = len(set(attractors_clusters)) * int(
                    lastp_for_truncate[15:]
                )
            fig = plt.figure(figsize=(25, 10))
            dn = hierarchy.dendrogram(
                Z,
                p=lastp_for_truncate,
                truncate_mode="lastp",
                color_threshold=normalized_distance_threshold,
                distance_sort=True,
                no_labels=False,
                show_contracted=True,
                ax=None,
            )
            plt.xlabel("Attractors")
            plt.ylabel("Hamming Distance")
            plt.title(dataset_pathway)
            plt.show()
            plt.savefig(
                dendrogram_saving_pathway
                + dataset_pathway
                + "_merged_attractors_truncation_dendrogram_gk.pdf"
            )
            plt.clf()

        # group attractors within same clsuter
        map_indices_attractors_within_clusters = (
            {}
        )  # stores dict with keys being clusters ids and values being lists which include int formated attractors within that cluster for current dataset_network pair
        for each_attr_indx in range(len(attractors_clusters)):
            if (
                attractors_clusters[each_attr_indx]
                in map_indices_attractors_within_clusters
            ):
                map_indices_attractors_within_clusters[
                    attractors_clusters[each_attr_indx]
                ].append(
                    all_attractors_mapped_to_by_cells[dataset_pathway][each_attr_indx]
                )
            else:
                map_indices_attractors_within_clusters[
                    attractors_clusters[each_attr_indx]
                ] = []
                map_indices_attractors_within_clusters[
                    attractors_clusters[each_attr_indx]
                ].append(
                    all_attractors_mapped_to_by_cells[dataset_pathway][each_attr_indx]
                )

        # store map_indices_attractors_within_clusters for each dataset_network pair
        set_of_clusters_per_dataset[
            dataset_pathway
        ] = map_indices_attractors_within_clusters

    return set_of_clusters_per_dataset


############################################################
##### Function that merges attractors within a cluster #####
############################################################


def Merge_Similar_Attractors(
    set_of_clusters_per_dataset, loaded_dist_dfs, attractors_per_dtst_ntrk_saving_path
):
    # merge attractors within same cluster
    set_of_clusters_with_representative_attractors_per_dataset = {}
    for (
        dataset_pathway
    ) in set_of_clusters_per_dataset:  # go through each dataset__network
        # obtain distance matrix for each attractor and its distance from each cell
        distances_df = loaded_dist_dfs[
            dataset_pathway
        ]  # get distances csv for current dataset__network
        cells_list = (
            distances_df.index.tolist()
        )  # get list of cells for the current dataset
        print("Done obtaining distances matrix for " + dataset_pathway)

        representative_attractors_for_each_cluster = {}
        cluster_counter = 1
        for each_cluster in set_of_clusters_per_dataset[
            dataset_pathway
        ]:  # go through each cluster id in current dataset__network
            attractors_in_this_cluster = set_of_clusters_per_dataset[dataset_pathway][
                each_cluster
            ]
            # First calculate distance between attractors within this cluster and cells
            cell_counter_for_attractors = [0] * len(
                attractors_in_this_cluster
            )  # counts number of cells mapping to each attractor within current cluster
            for each_cell in cells_list:
                distance_of_this_cell_from_each_attractor = [0] * len(
                    attractors_in_this_cluster
                )
                for each_attrac_indx in range(len(attractors_in_this_cluster)):
                    distance_attrac_cell = distances_df.loc[
                        each_cell, str(attractors_in_this_cluster[each_attrac_indx])
                    ]
                    distance_of_this_cell_from_each_attractor[
                        each_attrac_indx
                    ] = distance_attrac_cell
                min_index = distance_of_this_cell_from_each_attractor.index(
                    min(distance_of_this_cell_from_each_attractor)
                )  # returns first occurance of minimum value
                cell_counter_for_attractors[min_index] += 1
            chosen_attrac_indx = cell_counter_for_attractors.index(
                max(cell_counter_for_attractors)
            )
            representative_attractor = attractors_in_this_cluster[chosen_attrac_indx]
            # store cluster id and corresponding int formatted representative attractor
            representative_attractors_for_each_cluster[
                each_cluster
            ] = representative_attractor
            print(
                "Calculated Representative Attractor for Cluster #"
                + str(cluster_counter)
                + " Out Of "
                + str(len(set_of_clusters_per_dataset[dataset_pathway]))
                + " Clusters",
                end="\r",
            )
            cluster_counter += 1
        print(
            "Calculated Representative Attractor for Cluster #"
            + str(len(set_of_clusters_per_dataset[dataset_pathway]))
            + " Out Of "
            + str(len(set_of_clusters_per_dataset[dataset_pathway]))
            + " Clusters",
            end="\r",
        )

        set_of_clusters_with_representative_attractors_per_dataset[
            dataset_pathway
        ] = representative_attractors_for_each_cluster  # stores dataset__network as keys and values being a dictionary that has keys which are clusters' ids and values which are int formatted representative attractor
        print("Done Finding Representative Attractors in Each Cluster")

    # stores a csv with columns as cluster ids and rows being 0/1 values of representative attractors
    for dtst_ntrk in set_of_clusters_with_representative_attractors_per_dataset:
        df_dtst_ntrk = pd.DataFrame.from_dict(
            set_of_clusters_with_representative_attractors_per_dataset[dtst_ntrk]
        )
        df_dtst_ntrk.to_csv(
            attractors_per_dtst_ntrk_saving_path + dtst_ntrk + ".csv", index=False
        )

    return set_of_clusters_with_representative_attractors_per_dataset


###########################################################
##### Function that generates bar and frequency plots #####
###########################################################


def Create_Bar_Plot_And_Frequency_Plot(
    set_of_clusters_with_representative_attractors_per_dataset,
    loaded_dist_dfs,
    plots_saving_path,
    csv_saving_path,
    attractors_representing_clusters_indices_to_plot,
    datasets_cells_labels,
    restrict_attractors_plotted=False,
    Combined_Plot=True,
    bar_box_plots_fig_size=(3, 2),
):
    for (
        dataset_pathway
    ) in (
        set_of_clusters_with_representative_attractors_per_dataset
    ):  # go through each dataset_network pair
        representative_attractors_for_each_cluster = (
            set_of_clusters_with_representative_attractors_per_dataset[dataset_pathway]
        )  # stores dict of cluster ids and values are representative attractors
        dist_pd = loaded_dist_dfs[
            dataset_pathway
        ]  # get distances csv for current dataset__network
        cells_list = dist_pd.index.tolist()  # get list of cells for the current dataset

        Representative_Attractors = []  # representative attractors
        Representative_Attractors_Clusters_Labels = (
            []
        )  # the cluster id which has the representative attractor
        for representative_attrac in representative_attractors_for_each_cluster:
            Representative_Attractors.append(
                representative_attractors_for_each_cluster[representative_attrac]
            )
            Representative_Attractors_Clusters_Labels.append(representative_attrac)
        print(Representative_Attractors_Clusters_Labels)

        num_nodes_in_ntrk = len(
            Representative_Attractors[0]
        )  # stores number of nodes in this network
        print("num_nodes_in_ntrk: " + str(num_nodes_in_ntrk))

        dtst_name = dataset_pathway[
            0 : (dataset_pathway.find("__"))
        ]  # stores name of dataset
        # Store attractors and their clusters for current network
        ntrk_attractors_list = []
        ntrk_attractors_labels = []
        for cluster_id in set_of_clusters_with_representative_attractors_per_dataset[
            dataset_pathway
        ]:
            ntrk_attractors_labels.append(cluster_id)
            ntrk_attractors_list.append(
                set_of_clusters_with_representative_attractors_per_dataset[
                    dataset_pathway
                ][cluster_id]
            )
        print(ntrk_attractors_labels)
        # get labels of cell types for the datasett
        label1_txt = datasets_cells_labels[dtst_name]["Label1"]
        label2_txt = datasets_cells_labels[dtst_name]["Label2"]
        # calculate distances between each type of cells and attractors
        label1_cells_counter_for_attractors = [0] * len(ntrk_attractors_list)
        label2_cells_counter_for_attractors = [0] * len(ntrk_attractors_list)
        label1_cells_distances_from_attractors = {}
        label2_cells_distances_from_attractors = {}
        # initialize distances dictionaries
        for attr_lbl in ntrk_attractors_labels:
            label1_cells_distances_from_attractors[attr_lbl] = []
            label2_cells_distances_from_attractors[attr_lbl] = []

        # to store cells' names and distances for each attractors
        label1_cells_list_for_attractors = {}
        label2_cells_list_for_attractors = {}
        for attr_id in ntrk_attractors_labels:
            label1_cells_list_for_attractors[attr_id] = []
            label2_cells_list_for_attractors[attr_id] = []

        for each_cell in cells_list:
            distance_of_this_cell_from_each_attractor = [0] * len(ntrk_attractors_list)
            for each_attrac_indx in range(len(ntrk_attractors_list)):
                distance_attrac_cell = dist_pd.loc[
                    each_cell, str(ntrk_attractors_list[each_attrac_indx])
                ]
                distance_of_this_cell_from_each_attractor[
                    each_attrac_indx
                ] = distance_attrac_cell
            min_dist_value = min(distance_of_this_cell_from_each_attractor)
            min_index = distance_of_this_cell_from_each_attractor.index(
                min(distance_of_this_cell_from_each_attractor)
            )  # returns first occurance of minimum value
            if label1_txt in each_cell:
                label1_cells_counter_for_attractors[min_index] += 1
                if (
                    ntrk_attractors_labels[min_index]
                    not in label1_cells_distances_from_attractors
                ):
                    label1_cells_distances_from_attractors[
                        ntrk_attractors_labels[min_index]
                    ] = []
                label1_cells_distances_from_attractors[
                    ntrk_attractors_labels[min_index]
                ].append(min_dist_value)
                label1_cells_list_for_attractors[
                    ntrk_attractors_labels[min_index]
                ].append(
                    [each_cell, min_dist_value]
                )  # store cell names and DISTANCES BETWEEN CELL AND ATTRACTOR
            elif label2_txt in each_cell:
                label2_cells_counter_for_attractors[min_index] += 1
                if (
                    ntrk_attractors_labels[min_index]
                    not in label2_cells_distances_from_attractors
                ):
                    label2_cells_distances_from_attractors[
                        ntrk_attractors_labels[min_index]
                    ] = []
                label2_cells_distances_from_attractors[
                    ntrk_attractors_labels[min_index]
                ].append(min_dist_value)
                label2_cells_list_for_attractors[
                    ntrk_attractors_labels[min_index]
                ].append(
                    [each_cell, min_dist_value]
                )  # store cell names and DISTANCES BETWEEN CELL AND ATTRACTOR
            else:
                print("Could Not Identify Identity Of Cell!")
        # Calculate percentages from counts
        attracs_list = tuple(ntrk_attractors_list)
        label1_cells_counter = tuple(label1_cells_counter_for_attractors)
        label2_cells_counter = tuple(label2_cells_counter_for_attractors)
        sum_label1_cells = sum(label1_cells_counter)
        sum_label2_cells = sum(label2_cells_counter)
        print("label1: " + str(sum_label1_cells))
        print("label2: " + str(sum_label2_cells))
        label1_cells_percentages = [
            (i1 / sum_label1_cells) * 100 for i1 in label1_cells_counter
        ]
        label2_cells_percentages = [
            (i2 / sum_label2_cells) * 100 for i2 in label2_cells_counter
        ]
        print(label1_cells_percentages)
        print(label2_cells_percentages)
        for i in label1_cells_distances_from_attractors:
            if len(label1_cells_distances_from_attractors[i]) < 1:
                print("No Cells Mapped to Attractor " + str(i))
                print(
                    "Adding a nan value to this attractor's distances for it to still be plotted"
                )
                label1_cells_distances_from_attractors[i].append(np.nan)
                print(label1_cells_distances_from_attractors[i])
        print("\n\n")
        for i in label2_cells_distances_from_attractors:
            if len(label2_cells_distances_from_attractors[i]) < 1:
                print("No Cells Mapped to Attractor " + str(i))
                print(
                    "Adding a nan value to this attractor's distances for it to still be plotted"
                )
                label2_cells_distances_from_attractors[i].append(np.nan)
                print(label2_cells_distances_from_attractors[i])

        # pickle cells' names for each attracctor for this dataset_pathway as dictionaries
        
        with open(
            csv_saving_path
            + "Label1_"
            + label1_txt
            + "_Dataset_Pathway_"
            + dataset_pathway
            + "_cells_name_in_attractors.pickle",
            "wb",
        ) as handle1:
            pickle.dump(
                label1_cells_list_for_attractors,
                handle1,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            csv_saving_path
            + "Label2_"
            + label2_txt
            + "_Dataset_Pathway_"
            + dataset_pathway
            + "_cells_name_in_attractors.pickle",
            "wb",
        ) as handle2:
            pickle.dump(
                label2_cells_list_for_attractors,
                handle2,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            csv_saving_path
            + "num_nodes_in_ntrk_Dataset_Pathway_"
            + dataset_pathway
            + ".pickle",
            "wb",
        ) as handle3:
            pickle.dump(num_nodes_in_ntrk, handle3, protocol=pickle.HIGHEST_PROTOCOL)
        
        # chi-sqaured test
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
            # create box plots and frequency plots
            # for boxplots
            labels_list_for_all_repr_attr = []
            distances_list_for_all_repr_attr = []
            cell_type_list_for_all_repr_attr = []
            ##### LABEL1 #####
            for each_repr_attr in label1_cells_distances_from_attractors:
                labels_list_for_repr_attr = [each_repr_attr] * len(
                    label1_cells_distances_from_attractors[each_repr_attr]
                )
                labels_list_for_all_repr_attr = (
                    labels_list_for_all_repr_attr + labels_list_for_repr_attr
                )
                cell_type_list_for_repr_attr = [label1_txt] * len(
                    label1_cells_distances_from_attractors[each_repr_attr]
                )
                cell_type_list_for_all_repr_attr = (
                    cell_type_list_for_all_repr_attr + cell_type_list_for_repr_attr
                )
                distances_list_for_all_repr_attr = (
                    distances_list_for_all_repr_attr
                    + label1_cells_distances_from_attractors[each_repr_attr]
                )
            ##### LABEL2 #####
            for each_repr_attr_L2 in label2_cells_distances_from_attractors:
                labels_list_for_repr_attr_L2 = [each_repr_attr_L2] * len(
                    label2_cells_distances_from_attractors[each_repr_attr_L2]
                )
                labels_list_for_all_repr_attr = (
                    labels_list_for_all_repr_attr + labels_list_for_repr_attr_L2
                )
                cell_type_list_for_repr_attr_L2 = [label2_txt] * len(
                    label2_cells_distances_from_attractors[each_repr_attr_L2]
                )
                cell_type_list_for_all_repr_attr = (
                    cell_type_list_for_all_repr_attr + cell_type_list_for_repr_attr_L2
                )
                distances_list_for_all_repr_attr = (
                    distances_list_for_all_repr_attr
                    + label2_cells_distances_from_attractors[each_repr_attr_L2]
                )
            ##### BOTH LABELS #####
            distances_list_for_all_repr_attr = list(
                (np.array(distances_list_for_all_repr_attr) / num_nodes_in_ntrk) * 100
            )
            df_for_plotting = pd.DataFrame.from_dict(
                {
                    "representative_attractor_label": labels_list_for_all_repr_attr,
                    "distances_from_mapped_cells": distances_list_for_all_repr_attr,
                    "cell_type_label": cell_type_list_for_all_repr_attr,
                }
            )
            sns.set_context(
                "paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8}
            )

            # for frequency plots
            ##### LABEL1 #####
            Y5_L1 = [round(i, 1) for i in label1_cells_percentages]
            X5_L1 = (
                np.array(ntrk_attractors_labels) - 0.2 - 1
            )  # -1 just to fit on the axis created by sns for the boxplot. For the correct attactors' labels see other axis and not this one.
            ##### LABEL2 #####
            Y5_L2 = [round(i, 1) for i in label2_cells_percentages]
            X5_L2 = (
                np.array(ntrk_attractors_labels) + 0.2 - 1
            )  # -1 just to fit on the axis created by sns for the boxplot. For the correct attactors' labels see other axis and not this one.

            # combined
            fig = plt.figure(figsize=(bar_box_plots_fig_size))
            grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
            ax_main = fig.add_subplot(grid[1:4, 0:4])
            ax_top = fig.add_subplot(grid[0:1, 0:4])
            sns.boxplot(
                x="representative_attractor_label",
                y="distances_from_mapped_cells",
                data=df_for_plotting,
                ax=ax_main,
                hue="cell_type_label",
                showfliers=False,
                palette={label1_txt: "#cecfcf", label2_txt: "#252525"},
            )  # palette=["tab:blue"]
            # sns.stripplot(x="representative_attractor_label", y="distances_from_mapped_cells",data=df_for_plotting, ax=ax_main, jitter=True, marker='o', alpha=0.5, color='black')
            # color was tab:blue and tab:orange
            bar_container_L1 = ax_top.bar(
                X5_L1, Y5_L1, 0.4, color="#cecfcf", label=label1_txt, edgecolor="black"
            )
            bar_container_L2 = ax_top.bar(
                X5_L2, Y5_L2, 0.4, color="#252525", label=label2_txt, edgecolor="black"
            )
            ax_top.get_shared_x_axes().join(ax_top, ax_main)
            ax_top.set(title=dataset_pathway, xlabel="", ylabel="Percentage\nof Cells")
            ax_top.set_ylabel(
                "Percentage\nof Cells", rotation=90, fontsize=8, labelpad=5
            )
            ax_top.set_title(dataset_pathway, fontsize=8, pad=5)
            # ax_top.set_xticks(list(range(1,len(X5)+1)))
            ax_top.set_xticks([])
            # ax_top.set_xticklabels(list(range(1,len(X5)+1)))
            plt.setp(ax_top.get_xticklabels(), visible=False)
            # ax_top.grid(color='gray',axis='y',alpha=0.3)
            ax_top.bar_label(bar_container_L1, label_type="edge", size=7)
            ax_top.bar_label(bar_container_L2, label_type="edge", size=7)
            max_y_top = int(max(Y5_L1 + Y5_L2))
            min_y_top = int(min(Y5_L1 + Y5_L2))
            step_y_top = 20
            ax_top.set_yticks(list(range(0, 121, step_y_top)))
            ax_top.set_yticklabels(
                [str(i) for i in list(range(0, 101, step_y_top))] + [""], fontsize=7
            )
            ax_main.set(
                xlabel="Representative Attractors",
                ylabel="Normalized Distances\nFrom Mapped Cells",
            )
            ax_main.set_ylabel(
                "Normalized Distances\nFrom Mapped Cells",
                rotation=90,
                fontsize=8,
                labelpad=5,
            )
            ax_main.set_xlabel("Representative Attractors", fontsize=8, labelpad=5)
            max_y = int(max(list(df_for_plotting["distances_from_mapped_cells"])))
            min_y = int(min(list(df_for_plotting["distances_from_mapped_cells"])))
            if (max_y - min_y) < 20:
                step_y = 2
            else:
                step_y = 5
            ax_main.set_yticks(list(range(min_y, max_y + 1, step_y)))
            ax_main.set_yticklabels(list(range(min_y, max_y + 1, step_y)))
            ax_main.grid(color="gray", axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.savefig(plots_saving_path + dataset_pathway + "_combined_plot.pdf")
            plt.clf()

