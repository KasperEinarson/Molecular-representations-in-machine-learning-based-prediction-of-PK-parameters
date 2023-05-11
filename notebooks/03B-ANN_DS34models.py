''' 
Notebook that takes the splittet data from "Split_data.ipynb" and model the data using CNN layers for both sequential input.
Thus DS34, DS1234 and DS134 and DS234 are being processed in this notebook
Output: RMSE on the testsets exported for visualization in "Figures.ipynb"

'''

import sys, os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'src','insulin_pk','utils')) 
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
Full_data = pd.read_csv(os.path.join(ROOT_DIR, 'data','processed', 'full_data_set.csv'))
Full_data.set_index("nncno",inplace=True)
## Load data in folds and select only relevant descriptorset:
DS1_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS12_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS2_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS3_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS4_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))


for i in range(len(DS4_folds)):
    for j in range(3):
        DS12_folds[i][j] = contruct_descriptor_sets(DS12_folds[i][j],pd.Series("12"))
        DS1_folds[i][j] = contruct_descriptor_sets(DS1_folds[i][j],pd.Series("1"))
        DS2_folds[i][j] = contruct_descriptor_sets(DS2_folds[i][j],pd.Series("2"))
        DS3_folds[i][j] = contruct_descriptor_sets(DS3_folds[i][j],pd.Series("3"))
        DS4_folds[i][j] = contruct_descriptor_sets(DS4_folds[i][j],pd.Series("4"))


PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']

# Set training/validation epochs and number of bayesian optimization rounds
EPOCH = 200
N_TRIALS = 30

# Loop over test folds DS1234 =======================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS12_folds[i][0].astype(np.float64),DS12_folds[i][3].astype(np.float32)  
    X_val,Y_val = DS12_folds[i][1].astype(np.float64),DS12_folds[i][4].astype(np.float32)
    X_test,Y_test = DS12_folds[i][2].astype(np.float64),DS12_folds[i][5].astype(np.float32)
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    X_train_smiles = DS4_folds[i][0].astype(np.float64)
    X_train_smiles.set_index(X_train.index,inplace = True)
    X_val_smiles = DS4_folds[i][1].astype(np.float64)
    X_val_smiles.set_index(X_val.index,inplace = True)
    X_test_smiles = DS4_folds[i][2].astype(np.float64)
    X_test_smiles.set_index(X_test.index,inplace = True)
    
    dataset_train = Dataset_all_conc(X_train,X_train_seq,X_train_smiles,Y_train)
    dataset_val = Dataset_all_conc(X_val,X_val_seq,X_val_smiles,Y_val)
    dataset_test = Dataset_all_conc(X_test,X_test_seq,X_test_smiles,Y_test)
    
    # Hyper parameter optimization.
    study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial:objective_DS1234(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS1234_fold{0}.pt'.format(i+1)),
                                              input_dim_desc = X_train.shape[1]),
                                              n_trials = N_TRIALS,
                                              n_jobs=-1)
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS1234_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS1234_fold{0}.pt'.format(i)),'rb'))
    
  
    
    # Build model using best hyperparameters:
    
    model = model_DS1234_build(best_params,input_dim_desc = X_train.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1234_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    Vivo_train_results = train_and_validate(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS1234_fold{0}.pt'.format(i+1)),
                                           trial = "None")
    
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS1234_fold{0}.pt'.format(i+1)),Y_data_for_index = Y_test)
   
    CV_folds_test_vivo[i] = Vivo_test_results
    
    model.apply(reset_weights)

with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS1234.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )




# Loop over test folds DS134 =======================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS1_folds[i][0].astype(np.float64),DS1_folds[i][3].astype(np.float32)  
    X_val,Y_val = DS1_folds[i][1].astype(np.float64),DS1_folds[i][4].astype(np.float32)
    X_test,Y_test = DS1_folds[i][2].astype(np.float64),DS1_folds[i][5].astype(np.float32)
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    X_train_smiles = DS4_folds[i][0].astype(np.float64)
    X_train_smiles.set_index(X_train.index,inplace = True)
    X_val_smiles = DS4_folds[i][1].astype(np.float64)
    X_val_smiles.set_index(X_val.index,inplace = True)
    X_test_smiles = DS4_folds[i][2].astype(np.float64)
    X_test_smiles.set_index(X_test.index,inplace = True)
    
    dataset_train = Dataset_all_conc(X_train,X_train_seq,X_train_smiles,Y_train)
    dataset_val = Dataset_all_conc(X_val,X_val_seq,X_val_smiles,Y_val)
    dataset_test = Dataset_all_conc(X_test,X_test_seq,X_test_smiles,Y_test)
    
    # Hyper parameter optimization.
    study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial:objective_DS1234(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS134_fold{0}.pt'.format(i+1)),
                                              input_dim_desc = X_train.shape[1]), n_jobs=-1,
                                              n_trials = N_TRIALS)
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS134_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS134_fold{0}.pt'.format(i)),'rb'))
    
   
    # Build model using best hyperparameters:
    
    model = model_DS1234_build(best_params,input_dim_desc = X_train.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1234_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    Vivo_train_results = train_and_validate(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS134_fold{0}.pt'.format(i+1)),
                                           trial = "None")
    
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS134_fold{0}.pt'.format(i+1)),Y_data_for_index = Y_test)
   
    CV_folds_test_vivo[i] = Vivo_test_results
    
    model.apply(reset_weights)

with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS134.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )

# Loop over test folds DS234 =======================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS2_folds[i][0].astype(np.float64),DS2_folds[i][3].astype(np.float32)  
    X_val,Y_val = DS2_folds[i][1].astype(np.float64),DS2_folds[i][4].astype(np.float32)
    X_test,Y_test = DS2_folds[i][2].astype(np.float64),DS2_folds[i][5].astype(np.float32)
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    X_train_smiles = DS4_folds[i][0].astype(np.float64)
    X_train_smiles.set_index(X_train.index,inplace = True)
    X_val_smiles = DS4_folds[i][1].astype(np.float64)
    X_val_smiles.set_index(X_val.index,inplace = True)
    X_test_smiles = DS4_folds[i][2].astype(np.float64)
    X_test_smiles.set_index(X_test.index,inplace = True)
    
    dataset_train = Dataset_all_conc(X_train,X_train_seq,X_train_smiles,Y_train)
    dataset_val = Dataset_all_conc(X_val,X_val_seq,X_val_smiles,Y_val)
    dataset_test = Dataset_all_conc(X_test,X_test_seq,X_test_smiles,Y_test)
    
    # Hyper parameter optimization.
    study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial:objective_DS1234(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS234_fold{0}.pt'.format(i+1)),
                                              input_dim_desc = X_train.shape[1]), n_jobs=-1,
                                              n_trials = N_TRIALS)
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS234_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS234_fold{0}.pt'.format(i)),'rb'))
    
    
    
    # Build model using best hyperparameters:
    
    model = model_DS1234_build(best_params,input_dim_desc = X_train.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1234_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    Vivo_train_results = train_and_validate(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS234_fold{0}.pt'.format(i+1)),
                                           trial = "None")
    
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS234_fold{0}.pt'.format(i+1)),Y_data_for_index = Y_test)
   
    CV_folds_test_vivo[i] = Vivo_test_results
    
    model.apply(reset_weights)

with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS234.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )


# Loop over test folds DS34 =======================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    X_train, Y_train = DS2_folds[i][0].astype(np.float64),DS2_folds[i][3].astype(np.float32)  
    X_val,Y_val = DS2_folds[i][1].astype(np.float64),DS2_folds[i][4].astype(np.float32)
    X_test,Y_test = DS2_folds[i][2].astype(np.float64),DS2_folds[i][5].astype(np.float32)
    
    ## Set all numeric entries to 0 (as we only have sequential data in this case). Ineffective but valid if we want to keep using the current framework and functions:
    X_train = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)
    X_val = pd.DataFrame(0.0, index=X_val.index, columns=X_val.columns)
    X_test = pd.DataFrame(0.0, index=X_test.index, columns=X_test.columns)
    
    
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    X_train_smiles = DS4_folds[i][0].astype(np.float64)
    X_train_smiles.set_index(X_train.index,inplace = True)
    X_val_smiles = DS4_folds[i][1].astype(np.float64)
    X_val_smiles.set_index(X_val.index,inplace = True)
    X_test_smiles = DS4_folds[i][2].astype(np.float64)
    X_test_smiles.set_index(X_test.index,inplace = True)
    
    dataset_train = Dataset_all_conc(X_train,X_train_seq,X_train_smiles,Y_train)
    dataset_val = Dataset_all_conc(X_val,X_val_seq,X_val_smiles,Y_val)
    dataset_test = Dataset_all_conc(X_test,X_test_seq,X_test_smiles,Y_test)
    
    # Hyper parameter optimization.
    study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial:objective_DS1234(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS34_fold{0}.pt'.format(i+1)),
                                              input_dim_desc = X_train.shape[1]),n_jobs=-1,
                                              n_trials = N_TRIALS)
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS34_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS34_fold{0}.pt'.format(i)),'rb'))
    
   
    # Build model using best hyperparameters:
    
    model = model_DS1234_build(best_params,input_dim_desc = X_train.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1234_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    Vivo_train_results = train_and_validate(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS34_fold{0}.pt'.format(i+1)),
                                           trial = "None")
    
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_FFNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path = os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS34_fold{0}.pt'.format(i+1)),Y_data_for_index = Y_test)
   
    CV_folds_test_vivo[i] = Vivo_test_results
    
    model.apply(reset_weights)

with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS34.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )

