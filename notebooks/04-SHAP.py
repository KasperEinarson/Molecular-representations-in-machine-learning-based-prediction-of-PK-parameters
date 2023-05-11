#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' 
Calculates SHAP values on ANN using DS1234. Best hyperparameters are found during nested CV

'''

import sys, os
sys.path.append('../src/insulin_pk/utils/') 
import pickle  
import torch
import optuna
import random
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import warnings
from torch import nn
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from itertools import repeat, chain
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import seaborn as sns
import shap
# Import own modules:
from utils import *
# Supress optuna outputs and torch userwarnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Load full data (for SHAP analysis)
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']
get_ipython().system('jupyter nbconvert --to script "04-SHAP.ipynb"')


# In[2]:


optuna_hyp = {}
for i in range(5):
    tmp = pickle.load(open('../models/Optuna_DS1234_fold{0}.pt'.format(i),'rb'))
    optuna_hyp[i] = tmp
#pd.DataFrame(optuna_hyp)


# In[3]:


best_median_params = pd.DataFrame(optuna_hyp).median(1).to_dict()
# Dumb way to change the values to int..
best_median_params["Batch_Size"] = int(best_median_params["Batch_Size"])
best_median_params["conv1_filters_DS3"] = int(best_median_params["conv1_filters_DS3"])
best_median_params["conv2_filters_DS3"] = int(best_median_params["conv2_filters_DS3"])
best_median_params["conv1_filters_DS4"] = int(best_median_params["conv1_filters_DS4"])
best_median_params["conv2_filters_DS4"]= int(best_median_params["conv2_filters_DS4"])
best_median_params["Kernel_size1_DS3"] = int(best_median_params["Kernel_size1_DS3"])
best_median_params["Kernel_size1_DS4"] = int(best_median_params["Kernel_size1_DS4"])
best_median_params["FC_after_CNN_DS3"] = int(best_median_params["FC_after_CNN_DS3"])
best_median_params["FC_after_CNN_DS4"] = int(best_median_params["FC_after_CNN_DS4"])
best_median_params["FC_After_DS12"] = int(best_median_params["FC_After_DS12"])
best_median_params["FC_Concatenation"] = int(best_median_params["FC_Concatenation"])
best_median_params


# In[4]:


## Split the full dataset in 80% Training, 20% Test
Data_1234 = contruct_descriptor_sets(Full_data,pd.Series("DS1234"),with_PK = True)
X = Data_1234.drop(PK_names,axis=1)
Y = Data_1234[PK_names]
Y = np.log(Y)
train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 42).split(X, groups=X.index.get_level_values("nncno")))
X_train = X.iloc[train_inds,:]
Y_train = Y.iloc[train_inds,:]
X_test = X.iloc[test_inds,:]
Y_test = Y.iloc[test_inds,:]
## Scale:
scaler_X = StandardScaler().fit(X_train)
scaler_Y = StandardScaler().fit(Y_train)
X_train_scaled = pd.DataFrame(scaler_X.transform(X_train),columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),columns=X_test.columns)
Y_train_scaled = pd.DataFrame(scaler_Y.transform(Y_train),columns=Y_train.columns).astype(np.float32)
Y_test_scaled = pd.DataFrame(scaler_Y.transform(Y_test),columns=Y_test.columns).astype(np.float32)
# Split data into individual components:
X_train_DS4 =  X_train_scaled[X_train_scaled.columns[pd.Series(X_train_scaled.columns).str.startswith('DS4_')]].astype(np.float32)
X_test_DS4 =  X_test_scaled[X_test_scaled.columns[pd.Series(X_test_scaled.columns).str.startswith('DS4_')]].astype(np.float32)
X_train_DS3 =  X_train_scaled[X_train_scaled.columns[pd.Series(X_train_scaled.columns).str.startswith('DS3_')]].astype(np.float32)
X_test_DS3 =  X_test_scaled[X_test_scaled.columns[pd.Series(X_test_scaled.columns).str.startswith('DS3_')]].astype(np.float32)
X_train_DS12 = X_train_scaled.loc[:, X_train_scaled.columns.str.startswith("DS1") | X_train_scaled.columns.str.startswith("DS2")].astype(np.float32)
X_test_DS12 = X_test_scaled.loc[:, X_test_scaled.columns.str.startswith("DS1") | X_test_scaled.columns.str.startswith("DS2")].astype(np.float32) 
## Dataloaders:
dataset_train = Dataset_all_conc(X_train_DS12,X_train_DS3,X_train_DS4,Y_train_scaled)
dataset_test = Dataset_all_conc(X_test_DS12,X_test_DS3,X_test_DS4,Y_test_scaled)
# Make model and train using (median) best hyperparameters from nested CV:

Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)


# In[5]:


# Takes quite a while to train.
#DS1234_train_results = train_and_validate(Params = best_median_params,Model = Full_DS1234_model,Data_train = dataset_train ,Data_Val = dataset_test,
#                                            scaler_Y = scaler_Y,EPOCH = 200,save_model = True,save_path = "../models/DS1234_SHAP.pt",
#                                          trial = "None")


# In[10]:


Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)

Full_data_set_scaled = pd.concat([X_train_scaled,X_test_scaled],axis=0)
Full_data_set_scaled_torch = Variable( torch.from_numpy( Full_data_set_scaled.to_numpy(dtype=np.float32) ) )

X_train_scaled_torch =  Variable( torch.from_numpy( X_train_scaled.to_numpy(dtype=np.float32) ) )
X_test_scaled_torch = Variable( torch.from_numpy(X_test_scaled.to_numpy(dtype=np.float32)) )

Full_DS1234_model.load_state_dict(torch.load("../models/DS1234_SHAP.pt"))


# In[21]:


# Following takes ~ 10 minutes.
e = shap.DeepExplainer(Full_DS1234_model, Full_data_set_scaled_torch)
shap_values = e.shap_values(Full_data_set_scaled_torch)   
with open('../data/processed/ANN_SHAP_values_full.pkl','wb') as f:pickle.dump(shap_values,f )
with open('../data/processed/ANN_SHAP_values_X_test_full.pkl','wb') as f:pickle.dump(pd.DataFrame(pd.DataFrame(X_test_scaled_torch.numpy(),columns = X_train.columns )),f )
with open('../data/processed/ANN_SHAP_values_Y_train_full.pkl','wb') as f:pickle.dump(Y_train,f )


# In[22]:


shap_values = pickle.load(open('../data/processed/ANN_SHAP_values_full.pkl','rb'))


# In[6]:


X_corr = pd.DataFrame(X_train_scaled,columns=X_train_scaled.columns).corr()
plt.figure(figsize=(3,15))
dissimilarity = 1 - abs(X_corr)
Z = linkage(squareform(dissimilarity), 'complete')
#Z = linkage(X_corr, 'complete')

dendrogram(Z, labels=X_train_scaled.columns, orientation='right', 
 leaf_rotation=0,color_threshold = 0.1);
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=8)
ax.tick_params(axis='y', which='major', labelsize=4)
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_Dendogram.png",bbox_inches = 'tight')


# In[17]:


labels = fcluster(Z, 0.1, criterion='distance')
#plt.savefig("/home/kyei/Project1/Figures/Dendogram_DS1234_0_2.png")
Labels_pd = pd.DataFrame(pd.Series(X_train_scaled.columns),columns=["Features"])
Labels_pd["Cluster"] = labels
Dict_cluster = {k: g["Features"].tolist() for k,g in Labels_pd.groupby("Cluster")}



revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))


def grouped_shap(shap_vals, features, groups):
    '''
    Function to sum up shapvalues within highly correlated groups (ref: Scott Lundberg https://medium.com/@scottmlundberg/good-question-6229a343819f)
    '''
    
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    
    return shap_grouped


pandas_cluster = pd.DataFrame(pd.Series(Dict_cluster, name='Features'))
pandas_cluster['liststring'] = pandas_cluster['Features'].apply(lambda x: ','.join(map(str, x)))
pandas_cluster['liststring'] = pandas_cluster['liststring'].str.slice(0,30)


# In[18]:


groupmap = revert_dict(Dict_cluster)
shap_Tdf = pd.DataFrame(Full_data_set_scaled, columns=pd.Index(Full_data_set_scaled.columns, name='features')).T
shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
pd_features_grouped = grouped_shap(Full_data_set_scaled,Full_data_set_scaled.columns,Dict_cluster)
pd_features_grouped.columns = pandas_cluster.liststring




# In[19]:


fig = plt.figure(figsize=(40, 20))
columns = 1
rows = 3
what_PK = 2
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 10
for i in range(1, 4):
    fig.add_subplot(rows, columns, i)
    pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
    shap_type = grouped_shap(shap_values[i-1], X_test_scaled.columns,Dict_cluster)
    shap_type.columns = pd_cluster["liststring"]
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    
    #f = plt.figure()
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = pd_features_grouped.columns,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,10),color_bar_label='Feature value (sum)')
    #plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
    plt.gca().set_xlim(-0.6, 0.6)
    plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.xlabel('xlabel', fontsize=label_size)
    #plt.figsize=(18, 40)
    plt.title("{}".format(names[i-1]),fontsize=label_size)
    plt.gca().set_xlabel('')
    if i % 3 == 0:
            plt.xlabel('xlabel', fontsize=label_size)
            plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
            plt.gca().set_xlabel('SHAP values')
    
    
    
    #plt.show()
#plt.savefig("../reports/figures/ANN_DS1234_shap_total.png",bbox_inches = 'tight')


# In[ ]:


columns = 1
rows = 3
what_PK = 0
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 10
fig.add_subplot(rows, columns, i)

pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
shap_type = grouped_shap(shap_values[what_PK], X_test_scaled.columns,Dict_cluster)
shap_type.columns = pd_cluster["liststring"]
newcmp = sns.color_palette("crest_r", as_cmap=True)

#f = plt.figure()
newcmp = sns.color_palette("crest_r", as_cmap=True)
shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = pd_features_grouped.columns,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,5),color_bar_label='Feature value (sum)')
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
plt.gca().set_xlim(-0.7, 0.7)
plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.xlabel('xlabel', fontsize=label_size)
#plt.figsize=(18, 40)
plt.title("{}".format(names[what_PK]),fontsize=label_size)
plt.gca().set_xlabel('')
#plt.savefig("../reports/figures/ANN_DS1234_shap_CL.png",bbox_inches = 'tight')


# # Repeats

# In[ ]:


SHAP_features_repeats_CL = {}
SHAP_features_repeats_T12 = {}
SHAP_features_repeats_MRT = {}

SHAP_features_repeats_CL = pd.DataFrame()
SHAP_features_repeats_T12 = pd.DataFrame()
SHAP_features_repeats_MRT = pd.DataFrame()
for jj in range(5):
    ## Split the full dataset in 80% Training, 20% Test
    Data_1234 = contruct_descriptor_sets(Full_data,pd.Series("DS1234"),with_PK = True)
    X = Data_1234.drop(PK_names,axis=1)
    Y = Data_1234[PK_names]
    Y = np.log(Y)
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = jj).split(X, groups=X.index.get_level_values("nncno")))
    X_train = X.iloc[train_inds,:]
    Y_train = Y.iloc[train_inds,:]
    X_test = X.iloc[test_inds,:]
    Y_test = Y.iloc[test_inds,:]
    ## Scale:
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    X_train_scaled = pd.DataFrame(scaler_X.transform(X_train),columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),columns=X_test.columns)
    Y_train_scaled = pd.DataFrame(scaler_Y.transform(Y_train),columns=Y_train.columns).astype(np.float32)
    Y_test_scaled = pd.DataFrame(scaler_Y.transform(Y_test),columns=Y_test.columns).astype(np.float32)
    # Split data into individual components:
    X_train_DS4 =  X_train_scaled[X_train_scaled.columns[pd.Series(X_train_scaled.columns).str.startswith('DS4_')]].astype(np.float32)
    X_test_DS4 =  X_test_scaled[X_test_scaled.columns[pd.Series(X_test_scaled.columns).str.startswith('DS4_')]].astype(np.float32)
    X_train_DS3 =  X_train_scaled[X_train_scaled.columns[pd.Series(X_train_scaled.columns).str.startswith('DS3_')]].astype(np.float32)
    X_test_DS3 =  X_test_scaled[X_test_scaled.columns[pd.Series(X_test_scaled.columns).str.startswith('DS3_')]].astype(np.float32)
    X_train_DS12 = X_train_scaled.loc[:, X_train_scaled.columns.str.startswith("DS1") | X_train_scaled.columns.str.startswith("DS2")].astype(np.float32)
    X_test_DS12 = X_test_scaled.loc[:, X_test_scaled.columns.str.startswith("DS1") | X_test_scaled.columns.str.startswith("DS2")].astype(np.float32) 
    ## Dataloaders:
    dataset_train = Dataset_all_conc(X_train_DS12,X_train_DS3,X_train_DS4,Y_train_scaled)
    dataset_test = Dataset_all_conc(X_test_DS12,X_test_DS3,X_test_DS4,Y_test_scaled)
    # Make model and train using (median) best hyperparameters from nested CV:

    Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                     conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)
    # 
    print("Train model for shap experiment: {0}".format(jj))
    DS1234_train_results = train_and_validate(Params = best_median_params,Model = Full_DS1234_model,Data_train = dataset_train ,Data_Val = dataset_test,
                                                scaler_Y = scaler_Y,EPOCH = 150,save_model = True,save_path = "../models/DS1234_SHAP{0}.pt".format(jj),
                                               trial = "None")
    Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)

    X_train_scaled_torch =  Variable( torch.from_numpy( X_train_scaled.to_numpy(dtype=np.float32) ) )
    X_test_scaled_torch = Variable( torch.from_numpy(X_test_scaled.to_numpy(dtype=np.float32)) )

    Full_DS1234_model.load_state_dict(torch.load("../models/DS1234_SHAP{0}.pt".format(jj)))

    e = shap.DeepExplainer(Full_DS1234_model, X_train_scaled_torch)
    shap_values = e.shap_values(X_test_scaled_torch) 
    
    shap_PK_save = {}

    for i in range(3):
        pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
        shap_grouped = grouped_shap(shap_values[i], X.columns,Dict_cluster)
        shap_grouped.columns = pd_cluster["liststring"]
        shap_PK_save[i] = shap_grouped
    
    top10_shap_features_CL = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[0]).mean(0))]).iloc[::-1][:10]
    top10_shap_features_T12 = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[1]).mean(0))]).iloc[::-1][:10]
    top10_shap_features_MRT = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[2]).mean(0))]).iloc[::-1][:10]
        
    SHAP_features_repeats_CL = pd.concat([SHAP_features_repeats_CL,top10_shap_features_CL],axis=0)
    SHAP_features_repeats_T12 = pd.concat([SHAP_features_repeats_T12,top10_shap_features_T12],axis=0)
    SHAP_features_repeats_MRT = pd.concat([SHAP_features_repeats_MRT,top10_shap_features_MRT],axis=0)
                                      


# In[ ]:


SHAP_features_repeats_CL.value_counts()
SHAP_features_repeats_T12.value_counts()
SHAP_features_repeats_MRT.value_counts()


# In[ ]:


# In[ ]:


fig = plt.figure(figsize=(40, 20))
columns = 1
rows = 3
what_PK = 2
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 10
for i in range(1, 4):
    fig.add_subplot(rows, columns, i)
    pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
    shap_type = grouped_shap(shap_values[i-1], X_test_scaled.columns,Dict_cluster)
    shap_type.columns = pd_cluster["liststring"]
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    
    #f = plt.figure()
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = features_names_with_repeats[i-1],class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,10),color_bar_label='Feature value (sum)')
    #plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
    plt.gca().set_xlim(-0.6, 0.6)
    plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.xlabel('xlabel', fontsize=label_size)
    #plt.figsize=(18, 40)
    plt.title("{}".format(names[i-1]),fontsize=label_size)
    plt.gca().set_xlabel('')
    if i % 3 == 0:
            plt.xlabel('xlabel', fontsize=label_size)
            plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
            plt.gca().set_xlabel('SHAP values')
    
    
    
    #plt.show()
plt.savefig("../reports/figures/Figure_S4.png",bbox_inches = 'tight')


# In[ ]:


columns = 1
rows = 3
what_PK = 0
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 13
pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
shap_type = grouped_shap(shap_values[0], X_test_scaled.columns,Dict_cluster)
shap_type.columns = pd_cluster["liststring"]
newcmp = sns.color_palette("crest_r", as_cmap=True)

#f = plt.figure()
newcmp = sns.color_palette("crest_r", as_cmap=True)
shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = features_names_with_repeats[0],class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (14,4),color_bar_label='Feature value (sum)')
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
plt.gca().set_xlim(-0.6, 0.6)
plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.xlabel('xlabel', fontsize=label_size)
#plt.figsize=(18, 40)
plt.title("{}".format(names[0]),fontsize=label_size)
plt.gca().set_xlabel('')
plt.savefig("../reports/figures/Figure6b.png",bbox_inches = 'tight',dpi = 500)


# # Repeat with SHAP values being the average over all molecules

# In[30]:


## Split the full dataset in 80% Training, 20% Test
Data_1234 = contruct_descriptor_sets(Full_data,pd.Series("DS1234"),with_PK = True)
X = Data_1234.drop(PK_names,axis=1)
Y = Data_1234[PK_names]
Y = np.log(Y)
## Scale:
scaler_X = StandardScaler().fit(X)
scaler_Y = StandardScaler().fit(Y)
X_scaled = pd.DataFrame(scaler_X.transform(X),columns=X.columns)
Y_scaled = pd.DataFrame(scaler_Y.transform(Y),columns=Y.columns).astype(np.float32)
# Split data into individual components:

X_scaled_DS4 =  X_scaled[X_scaled.columns[pd.Series(X_scaled.columns).str.startswith('DS4_')]].astype(np.float32)

X_scaled_DS3 =  X_scaled[X_scaled.columns[pd.Series(X_scaled.columns).str.startswith('DS3_')]].astype(np.float32)

X_scaled_DS12 = X_scaled.loc[:, X_scaled.columns.str.startswith("DS1") | X_scaled.columns.str.startswith("DS2")].astype(np.float32)

## Dataloaders:
dataset_train = Dataset_all_conc(X_scaled_DS12,X_scaled_DS3,X_scaled_DS4,Y_scaled)
# Make model and train using (median) best hyperparameters from nested CV:

Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)


# In[32]:


# Takes quite a while to train.
DS1234_train_results = train_and_validate(Params = best_median_params,Model = Full_DS1234_model,Data_train = dataset_train ,Data_Val = dataset_test,
                                            scaler_Y = scaler_Y,EPOCH = 200,save_model = True,save_path = "../models/DS1234_SHAP_full_data_shap.pt",
                                           trial = "None")


# In[33]:


Full_DS1234_model = model_DS1234_build(params = best_median_params, input_dim_desc = X_train_DS12.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1 ,padding_DS3 = 0 ,padding_DS4 = 0, max_pool_kernel_size3 = 3)


Full_data_set_scaled_torch = Variable( torch.from_numpy( X_scaled.to_numpy(dtype=np.float32) ) )

X_train_scaled_torch =  Variable( torch.from_numpy( X_scaled.to_numpy(dtype=np.float32) ) )

Full_DS1234_model.load_state_dict(torch.load("../models/DS1234_SHAP.pt"))

# Following takes ~ 10 minutes.
e = shap.DeepExplainer(Full_DS1234_model, Full_data_set_scaled_torch)
shap_values = e.shap_values(Full_data_set_scaled_torch)   
with open('../data/processed/ANN_SHAP_values_full_2.pkl','wb') as f:pickle.dump(shap_values,f )
with open('../data/processed/ANN_SHAP_values_X_test_full_2.pkl','wb') as f:pickle.dump(pd.DataFrame(pd.DataFrame(Full_data_set_scaled_torch.numpy(),columns = X_train.columns )),f )
with open('../data/processed/ANN_SHAP_values_Y_train_full_2.pkl','wb') as f:pickle.dump(Y_scaled,f )


# In[34]:


shap_values = pickle.load(open('../data/processed/ANN_SHAP_values_full_2.pkl','rb'))


# In[35]:


X_corr = pd.DataFrame(X_scaled,columns=X_scaled.columns).corr()
plt.figure(figsize=(3,15))
dissimilarity = 1 - abs(X_corr)
Z = linkage(squareform(dissimilarity), 'complete')
#Z = linkage(X_corr, 'complete')

dendrogram(Z, labels=X_train_scaled.columns, orientation='right', 
 leaf_rotation=0,color_threshold = 0.1);
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=8)
ax.tick_params(axis='y', which='major', labelsize=4)
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_Dendogram.png",bbox_inches = 'tight')


# In[36]:


labels = fcluster(Z, 0.1, criterion='distance')
#plt.savefig("/home/kyei/Project1/Figures/Dendogram_DS1234_0_2.png")
Labels_pd = pd.DataFrame(pd.Series(X_train_scaled.columns),columns=["Features"])
Labels_pd["Cluster"] = labels
Dict_cluster = {k: g["Features"].tolist() for k,g in Labels_pd.groupby("Cluster")}



revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))


def grouped_shap(shap_vals, features, groups):
    '''
    Function to sum up shapvalues within highly correlated groups (ref: Scott Lundberg https://medium.com/@scottmlundberg/good-question-6229a343819f)
    '''
    
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    
    return shap_grouped


pandas_cluster = pd.DataFrame(pd.Series(Dict_cluster, name='Features'))
pandas_cluster['liststring'] = pandas_cluster['Features'].apply(lambda x: ','.join(map(str, x)))
pandas_cluster['liststring'] = pandas_cluster['liststring'].str.slice(0,50)


# In[78]:


groupmap = revert_dict(Dict_cluster)
shap_Tdf = pd.DataFrame(X_scaled, columns=pd.Index(X_scaled.columns, name='features')).T
shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
pd_features_grouped = grouped_shap(Full_data_set_scaled,Full_data_set_scaled.columns,Dict_cluster)
pd_features_grouped.columns = pandas_cluster.liststring
pd_features_grouped.drop(["DS4_59","DS4_58","DS4_18","DS4_75","DS4_63","DS4_9,DS4_12,DS4_42,DS4_70,DS4_79,DS4_82",
                         "DS4_31,DS4_43,DS4_50,DS4_71,DS4_83,DS4_91","DS4_1,DS4_5,DS4_6,DS4_7,DS4_19,DS4_36,DS4_48,DS4_7",
                         "DS4_40"],axis=1,inplace=True)


# In[74]:


pd_features_grouped.loc[:, pd_features_grouped.columns.str.startswith('DS4_')].columns


# In[79]:


fig = plt.figure(figsize=(40, 20))
columns = 1
rows = 3
what_PK = 2
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 10
for i in range(1, 4):
    fig.add_subplot(rows, columns, i)
    pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
    shap_type = grouped_shap(shap_values[i-1], X_test_scaled.columns,Dict_cluster)
    shap_type.columns = pd_cluster["liststring"]
    shap_type.drop(["DS4_59","DS4_58","DS4_18","DS4_75","DS4_63","DS4_9,DS4_12,DS4_42,DS4_70,DS4_79,DS4_82",
                         "DS4_31,DS4_43,DS4_50,DS4_71,DS4_83,DS4_91","DS4_1,DS4_5,DS4_6,DS4_7,DS4_19,DS4_36,DS4_48,DS4_7","DS4_40"],axis=1,inplace=True)
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    
    #f = plt.figure()
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = pd_features_grouped.columns,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,10),color_bar_label='Feature value (sum)')
    #plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
    plt.gca().set_xlim(-1.5, 1.5)
    plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.xlabel('xlabel', fontsize=label_size)
    #plt.figsize=(18, 40)
    plt.title("{}".format(names[i-1]),fontsize=label_size)
    plt.gca().set_xlabel('')
    if i % 3 == 0:
            plt.xlabel('xlabel', fontsize=label_size)
            plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
            plt.gca().set_xlabel('SHAP values')
    
    
    
    #plt.show()
#plt.savefig("../reports/figures/ANN_DS1234_shap_total.png",bbox_inches = 'tight')


# In[ ]:


columns = 1
rows = 3
what_PK = 0
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 10
fig.add_subplot(rows, columns, i)

pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
shap_type = grouped_shap(shap_values[what_PK], X_test_scaled.columns,Dict_cluster)
shap_type.columns = pd_cluster["liststring"]
newcmp = sns.color_palette("crest_r", as_cmap=True)

#f = plt.figure()
newcmp = sns.color_palette("crest_r", as_cmap=True)
shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = pd_features_grouped.columns,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,5),color_bar_label='Feature value (sum)')
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
plt.gca().set_xlim(-0.7, 0.7)
plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.xlabel('xlabel', fontsize=label_size)
#plt.figsize=(18, 40)
plt.title("{}".format(names[what_PK]),fontsize=label_size)
plt.gca().set_xlabel('')
#plt.savefig("../reports/figures/ANN_DS1234_shap_CL.png",bbox_inches = 'tight')

