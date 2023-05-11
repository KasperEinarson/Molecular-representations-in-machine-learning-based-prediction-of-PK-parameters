#!/usr/bin/env python
# coding: utf-8

# In[2]:


''' 
Notebook that takes the splittet data from "Split_data.ipynb" and model the data using feed forward neural network on all (solely) numeric descriptors.
Thus only DS1, DS12 and DS2 are being processed in this notebook
Output: RMSE on the testsets exported for visualization in "Figures.ipynb"

'''

import sys, os
sys.path.append('../src/insulin_pk/utils/') 
import pickle  
import torch
import pickle
import seaborn as sns
import optuna
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import pandas as pd
import warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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

## Load data in folds and select only relevant descriptorset:
DS1_folds = pickle.load(open('../data/processed/Data_folds.pkl','rb'))
DS12_folds = pickle.load(open('../data/processed/Data_folds.pkl','rb'))
DS2_folds = pickle.load(open('../data/processed/Data_folds.pkl','rb'))

for i in range(len(DS1_folds)):
    for j in range(3):
        DS1_folds[i][j] = contruct_descriptor_sets(DS1_folds[i][j],pd.Series("1"))
        DS12_folds[i][j] = contruct_descriptor_sets(DS12_folds[i][j],pd.Series("12"))
        DS2_folds[i][j] = contruct_descriptor_sets(DS2_folds[i][j],pd.Series("2"))

PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']

EPOCHS = 200
N_TRIALS = 20
get_ipython().system('jupyter nbconvert --to script "03B-ANN_numeric.ipynb"')


# In[3]:


def plot_pca(X_train_numeric_scaled,X_sampled_from_test_numeric_scaled,fold):
    '''
    PCA plot on scaled (numeric) X data. 
        
    '''
    PCA_object= PCA(n_components=2)
    
    principalComponents_object = PCA_object.fit(X_train_numeric_scaled)
    principalComponents = principalComponents_object.transform(X_train_numeric_scaled)
    principalComponents_pd = pd.DataFrame(principalComponents,columns=["PC1","PC2"],index = X_train_numeric_scaled.index)
    Data_numeric_X_PCA = pd.concat([X_train_numeric_scaled,principalComponents_pd],axis=1)
    Data_numeric_X_PCA["Group"] = "Train"
    
    
    principalComponents_X_test = principalComponents_object.transform(X_sampled_from_test_numeric_scaled)
    principalComponents_pd_X_test = pd.DataFrame(principalComponents_X_test,columns=["PC1","PC2"],index = X_sampled_from_test_numeric_scaled.index)
    Data_numeric_X_test_PCA = pd.concat([X_sampled_from_test_numeric_scaled,principalComponents_pd_X_test],axis=1)
    Data_numeric_X_test_PCA["Group"] = "Test"

    Train_test_PCA_data = pd.concat([Data_numeric_X_PCA,Data_numeric_X_test_PCA],axis=0)
    

    plt.figure(figsize = (8,8))
    sns.scatterplot(data=Train_test_PCA_data, x="PC1", y="PC2",style="Group",hue = "Group", palette = ["black","blue"])
    plt.savefig("../reports/figures/X_train_val_vs_test{}_now.png".format(fold),bbox_inches = 'tight')
    
    

train_val_folds = [fold1_train_val,fold2_train_val,fold3_train_val,fold4_train_val,fold5_train_val]
test_folds = [DS12_folds[0][2],DS12_folds[1][2],DS12_folds[2][2],DS12_folds[3][2],DS12_folds[4][2]]


EMD_distances = []
for i in range(5):
    X_train = train_val_folds[i]
    X_test = test_folds[i]
    d = cdist(X_train,X_test)
    assignment = linear_sum_assignment(d)
    EMD_distances.append( (d[assignment].sum() / X_train.shape[0]))
    EMD_distances_np = np.array(EMD_distances)
EMD_distances_pd = pd.DataFrame(EMD_distances_np , columns = ["EMD"])


# In[99]:


Y_fold1_total_long = pd.melt(fold1_Y_total.reset_index(), id_vars=['nncno','Group'], value_vars=['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']).set_index("nncno")
Y_fold2_total_long = pd.melt(fold2_Y_total.reset_index(), id_vars=['nncno','Group'], value_vars=['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']).set_index("nncno")
Y_fold3_total_long = pd.melt(fold3_Y_total.reset_index(), id_vars=['nncno','Group'], value_vars=['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']).set_index("nncno")
Y_fold4_total_long = pd.melt(fold4_Y_total.reset_index(), id_vars=['nncno','Group'], value_vars=['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']).set_index("nncno")
Y_fold5_total_long = pd.melt(fold5_Y_total.reset_index(), id_vars=['nncno','Group'], value_vars=['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']).set_index("nncno")


# In[106]:


g = sns.FacetGrid(Y_fold5_total_long, col="variable",
                  height=8, aspect=1,sharey = False,hue = "Group")
g.map(sns.kdeplot, "value")
g.add_legend()
g.savefig("../reports/figures/Y_fold_train_test.png",bbox_inches = 'tight')


# In[83]:


sns.kdeplot(data = fold1_Y_total["T1/2[h]"])



# # DS1 using ANN

# In[5]:


# Loop over test folds DS1
CV_folds_test_vivo = {}
for i in range(len(DS1_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS1_folds[i][0],DS1_folds[i][3]  
    X_val,Y_val = DS1_folds[i][1],DS1_folds[i][4]
    X_test,Y_test = DS1_folds[i][2],DS1_folds[i][5]
    scaler_Y = pickle.load(open('../data/processed/Scaler_Y_{0}.pkl'.format(i),'rb'))
    
    dataset_train = Dataset_FFNN(X_train,Y_train)
    dataset_val = Dataset_FFNN(X_val,Y_val)  
    dataset_test = Dataset_FFNN(X_test,Y_test)
    
    
    
    ## Hyperparameter tuning section. Will only be needed once --------------------------------------------
    #study = optuna.create_study(direction="minimize")
    #study.optimize(lambda trial:objective_DS1(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
    #                                          Scaler_Y = scaler_Y,epoch=EPOCHS,save_model=False,save_path = "../models/VIVO_FFNN_DS1_fold{0}.pt".format(i+1)),
    #                                          n_trials = N_TRIALS)
    #trial_ = study.best_trial
    #with open('../models/Optuna_DS1_fold{0}.pkl'.format(i),'wb') as f:pickle.dump(trial_.params,f )
    #print("Best hyperparameters for fold {0} is saved".format(i+1))
    #print(f" best parameters for this fold {trial_.params}")
    
    
    ### ---------------------------------------------------------------------------
    best_params = pickle.load(open('../models/Optuna_DS1_fold{0}.pt'.format(i),'rb'))
   
    # Build model using best hyperparameters:
    model = build_model_DS1(best_params,in_features = X_train.shape[1])
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    print("Re-training best model")
    Vivo_train_results = train_and_validate_FFNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCHS = EPOCHS,save_model = True,save_path = "../models/VIVO_FFNN_DS1_fold{0}.pt".format(i+1))
    print("Model re-trained. Now testing")
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=Y_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(network = model,params = best_params,test_loader =  test_loader,scaler_Y = scaler_Y,save_path =  "../models/VIVO_FFNN_DS1_fold{0}.pt".format(i+1),Y_data_for_index = Y_test )
    print("Done testing")
    CV_folds_test_vivo[i] = Vivo_test_results
    # reset weights between each fold (maybe not nessersary as we also do this between the hyperparameters searches)
    model.apply(reset_weights)
with open("../data/processed/ANN_outer_5_test_DS1.pkl",'wb') as f:pickle.dump(CV_folds_test_vivo,f )
    


# # DS12 using ANN

# In[3]:


# Loop over test folds DS12
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS12_folds[i][0],DS12_folds[i][3]  
    X_val,Y_val = DS12_folds[i][1],DS12_folds[i][4]
    X_test,Y_test = DS12_folds[i][2],DS12_folds[i][5]
    scaler_Y = pickle.load(open('../data/processed/Scaler_Y_{0}.pkl'.format(i),'rb'))
    
    dataset_train = Dataset_FFNN(X_train,Y_train)
    dataset_val = Dataset_FFNN(X_val,Y_val)  
    dataset_test = Dataset_FFNN(X_test,Y_test)
    
    
    ## Hyperparameter tuning section. Will only be needed once --------------------------------------------
    study = optuna.create_study(direction="minimize")
    tudy.optimize(lambda trial:objective_DS1(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y = scaler_Y,epoch=EPOCHS,save_model=False,save_path = "../models/VIVO_FFNN_DS12_fold{0}.pt".format(i+1)),
                                              n_trials = N_TRIALS)
    trial_ = study.best_trial
    with open('../models/Optuna_DS12_fold{0}.pkl'.format(i),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    rint(f" best parameters for this fold {trial_.params}")
    
    
    ### ---------------------------------------------------------------------------
    best_params = pickle.load(open('../models/Optuna_DS12_fold{0}.pkl'.format(i),'rb'))
    
    # Build model using best hyperparameters:
    model = build_model_DS1(best_params,in_features = X_train.shape[1])
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    print("Re-training best model")
    Vivo_train_results = train_and_validate_FFNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCHS = EPOCHS,save_model = True,save_path = "../models/VIVO_FFNN_DS12_fold{0}.pt".format(i+1))
    print("Model re-trained. Now testing")
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=Y_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(network = model,params = best_params,test_loader =  test_loader,scaler_Y = scaler_Y,save_path =  "../models/VIVO_FFNN_DS12_fold{0}.pt".format(i+1),Y_data_for_index = Y_test)
    print("Done testing")
    CV_folds_test_vivo[i] = Vivo_test_results
    # reset weights between each fold (maybe not nessersary as we also do this between the hyperparameters searches)
    model.apply(reset_weights)
with open("../data/processed/ANN_outer_5_test_DS12.pkl",'wb') as f:pickle.dump(CV_folds_test_vivo,f )


# # DS2 using ANN

# In[8]:


# Loop over test folds DS2
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS2_folds[i][0],DS2_folds[i][3]  
    X_val,Y_val = DS2_folds[i][1],DS2_folds[i][4]
    X_test,Y_test = DS2_folds[i][2],DS2_folds[i][5]
    scaler_Y = pickle.load(open('../data/processed/Scaler_Y_{0}.pkl'.format(i),'rb'))
    
    dataset_train = Dataset_FFNN(X_train,Y_train)
    dataset_val = Dataset_FFNN(X_val,Y_val)  
    dataset_test = Dataset_FFNN(X_test,Y_test)
    
    
    ## Hyperparameter tuning section. Will only be needed once --------------------------------------------
    #study = optuna.create_study(direction="minimize")
    #study.optimize(lambda trial:objective_DS1(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
    #                                          Scaler_Y = scaler_Y,epoch=EPOCHS,save_model=False,save_path = "../models/VIVO_FFNN_DS2_fold{0}.pt".format(i+1)),
    #                                          n_trials = N_TRIALS)
    #trial_ = study.best_trial
    #with open('../models/Optuna_DS2_fold{0}.pkl'.format(i),'wb') as f:pickle.dump(trial_.params,f )
    #print("Best hyperparameters for fold {0} is saved".format(i+1))
    #print(f" best parameters for this fold {trial_.params}")
    
    
    ### ---------------------------------------------------------------------------
    best_params = pickle.load(open('../models/Optuna_DS2_fold{0}.pkl'.format(i),'rb'))
        
    # Build model using best hyperparameters:
    model = build_model_DS1(best_params,in_features = X_train.shape[1])
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    print("Re-training best model")
    Vivo_train_results = train_and_validate_FFNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCHS = EPOCHS,save_model = True,save_path = "../models/VIVO_FFNN_DS2_fold{0}.pt".format(i+1))
    print("Model re-trained. Now testing")
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=Y_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(network = model,params = best_params,test_loader =  test_loader,scaler_Y = scaler_Y,save_path =  "../models/VIVO_FFNN_DS2_fold{0}.pt".format(i+1),Y_data_for_index = Y_test)
    print("Done testing")
    CV_folds_test_vivo[i] = Vivo_test_results
    # reset weights between each fold (maybe not nessersary as we also do this between the hyperparameters searches)
    model.apply(reset_weights)
with open("../data/processed/ANN_outer_5_test_DS2.pkl",'wb') as f:pickle.dump(CV_folds_test_vivo,f )

