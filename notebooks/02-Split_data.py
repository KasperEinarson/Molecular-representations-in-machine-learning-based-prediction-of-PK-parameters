#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Notebook to split data into a user defined amount of folds for training, validation and test. This is done for the full descriptorset (DS1234) such that subsets of descriptors can be extracted in the model notebooks (RF, ANN).
Splitting the data here is only used for ANN model later on. The exact same splitting is done within "Random_Forest.ipynb" for RF model seperately.
Output: List of dataframes exported to folder "Processed data" which contains all relevant data splitted and scaled for each fold and ready as model (ANN) input.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
sys.path.append('../src/insulin_pk/utils/') 
from utils import *
get_ipython().system('jupyter nbconvert --to script "02-Split_data.ipynb"')


# In[2]:


Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]


# In[5]:


X = Full_data.drop(PK_names,axis=1)
Y = Full_data[PK_names]
Y = np.log(Y)


# Initialize empty list to fill up with train,val and test data for each fold for the full descriptorset
data_folds = []
# Split data into user defined cross validation folds:
# Note that this splitting is identical to the splitting in "Random Forest.ipynb" in order to be able to compare results between RF and ANN using paired t-test later on.
idx_outer = list(GroupKFold(n_splits = 5).split(Full_data,groups = Full_data.index.get_level_values("nncno")))
for i in range(len(idx_outer)): 
                X_train_outer, X_test, Y_train_outer, Y_test = X.iloc[idx_outer[i][0]], X.iloc[idx_outer[i][1]], Y.iloc[idx_outer[i][0]], Y.iloc[idx_outer[i][1]]
                # Start Inner split
                idx_inner = list(GroupShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42).split(X_train_outer,groups = X_train_outer.index.get_level_values("nncno")))[0]
                X_train, X_val, Y_train, Y_val = X_train_outer.iloc[idx_inner[0]], X_train_outer.iloc[idx_inner[1]], Y_train_outer.iloc[idx_inner[0]], Y_train_outer.iloc[idx_inner[1]]
                
                ## Scale data appropriately
                scaler_X = StandardScaler().fit(X_train)
                scaler_Y = StandardScaler().fit(Y_train)
                
                ### ------ scale X
                X_train_scaled = pd.DataFrame(scaler_X.transform(X_train),columns = X_train.columns)
                X_train_scaled = X_train_scaled.set_index(X_train.index)
                
                X_val_scaled = pd.DataFrame(scaler_X.transform(X_val),columns = X_val.columns)
                X_val_scaled = X_val_scaled.set_index(X_val.index)
    
                X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),columns = X_test.columns)
                X_test_scaled = X_test_scaled.set_index(X_test.index)
                ### ------ scale Y
                
                Y_train_scaled = pd.DataFrame(scaler_Y.transform(Y_train),index = Y_train.index,columns = PK_names)
                Y_val_scaled = pd.DataFrame(scaler_Y.transform(Y_val),index = Y_val.index,columns = PK_names)
                Y_test_scaled = pd.DataFrame(scaler_Y.transform(Y_test),index = Y_test.index,columns = PK_names)
                ### Append to list for export:
                data_folds.append([X_train_scaled,X_val_scaled,X_test_scaled,Y_train_scaled,Y_val_scaled,Y_test_scaled])
                ### Export Y scaler for later reconstruction of original PK values:
                with open('../data/processed/Scaler_Y_' + "{0}.pkl".format(i),'wb') as f:pickle.dump(scaler_Y,f )  

with open('../data/processed/Data_folds.pkl','wb') as f:pickle.dump(data_folds,f )             


# # Grouped data for ANN

# In[6]:


#Create data with groups
Data_with_groups = pd.read_excel("../data/raw/Data_with_groups.xlsx")
Data_with_groups.rename(columns={"NNCNo": "nncno"})
Data_with_groups.set_index("NNCNo",inplace=True)
Data_with_groups = Data_with_groups[~Data_with_groups.index.isin(["0148-0000-1247"])]
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
data_wg = pd.merge(Full_data, Data_with_groups["Groups"], left_index=True, right_index=True)
data_wg.index.name = "nncno"


# In[24]:


groups = ["Other", 'Acylation', 'No attachments', 'Concatenated proteins',
       'Antibody attachment (Fc)', 'AA extensions']     

for kk in groups:
    Full_data = data_wg[data_wg["Groups"] == kk]
    Full_data.drop("Groups",axis=1,inplace=True)
    X = Full_data.drop(PK_names,axis=1)
    Y = Full_data[PK_names]
    Y = np.log(Y)

    # Initialize empty list to fill up with train,val and test data for each fold for the full descriptorset
    data_folds = []
    # Split data into user defined cross validation folds:
    # Note that this splitting is identical to the splitting in "Random Forest.ipynb" in order to be able to compare results between RF and ANN using paired t-test later on.
    idx_outer = list(GroupKFold(n_splits = 5).split(Full_data,groups = Full_data.index.get_level_values("nncno")))
    for i in range(len(idx_outer)): 
                    X_train_outer, X_test, Y_train_outer, Y_test = X.iloc[idx_outer[i][0]], X.iloc[idx_outer[i][1]], Y.iloc[idx_outer[i][0]], Y.iloc[idx_outer[i][1]]
                    # Start Inner split
                    idx_inner = list(GroupShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42).split(X_train_outer,groups = X_train_outer.index.get_level_values("nncno")))[0]
                    X_train, X_val, Y_train, Y_val = X_train_outer.iloc[idx_inner[0]], X_train_outer.iloc[idx_inner[1]], Y_train_outer.iloc[idx_inner[0]], Y_train_outer.iloc[idx_inner[1]]

                    ## Scale data appropriately
                    scaler_X = StandardScaler().fit(X_train)
                    scaler_Y = StandardScaler().fit(Y_train)

                    ### ------ scale X
                    X_train_scaled = pd.DataFrame(scaler_X.transform(X_train),columns = X_train.columns)
                    X_train_scaled = X_train_scaled.set_index(X_train.index)

                    X_val_scaled = pd.DataFrame(scaler_X.transform(X_val),columns = X_val.columns)
                    X_val_scaled = X_val_scaled.set_index(X_val.index)

                    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),columns = X_test.columns)
                    X_test_scaled = X_test_scaled.set_index(X_test.index)
                    ### ------ scale Y

                    Y_train_scaled = pd.DataFrame(scaler_Y.transform(Y_train),index = Y_train.index,columns = PK_names)
                    Y_val_scaled = pd.DataFrame(scaler_Y.transform(Y_val),index = Y_val.index,columns = PK_names)
                    Y_test_scaled = pd.DataFrame(scaler_Y.transform(Y_test),index = Y_test.index,columns = PK_names)
                    ### Append to list for export:
                    data_folds.append([X_train_scaled,X_val_scaled,X_test_scaled,Y_train_scaled,Y_val_scaled,Y_test_scaled])
                    ### Export Y scaler for later reconstruction of original PK values:
                    with open('../data/processed/Scaler_Y_{}_{}.pkl'.format(kk,i),'wb') as f:pickle.dump(scaler_Y,f )  

    with open('../data/processed/Data_folds_{}.pkl'.format(kk),'wb') as f:pickle.dump(data_folds,f )             


# In[36]:


# Sanity check
for kk in groups:
    dat_tmp = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))
    print(kk)
    print(dat_tmp[0][0].shape[0] + dat_tmp[0][1].shape[0] + dat_tmp[0][2].shape[0])

