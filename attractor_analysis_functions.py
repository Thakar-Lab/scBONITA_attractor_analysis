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
