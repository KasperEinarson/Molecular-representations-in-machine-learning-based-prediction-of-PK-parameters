''' 
Notebook that takes the splittet data from "Split_data.ipynb" and model the data using feed forward neural network on descriptorset including DS3
Thus DS3, DS13 and DS23 and DS123 are being processed in this notebook
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
DS1_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS12_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS2_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))
DS3_folds = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Data_folds.pkl'),'rb'))


for i in range(len(DS3_folds)):
    for j in range(3):
        DS12_folds[i][j] = contruct_descriptor_sets(DS12_folds[i][j],pd.Series("12"))
        DS1_folds[i][j] = contruct_descriptor_sets(DS1_folds[i][j],pd.Series("1"))
        DS2_folds[i][j] = contruct_descriptor_sets(DS2_folds[i][j],pd.Series("2"))
        DS3_folds[i][j] = contruct_descriptor_sets(DS3_folds[i][j],pd.Series("3"))


PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']

# Set training/validation epochs and number of bayesian optimization rounds
EPOCH = 200
N_TRIALS = 30

# Loop over test folds DS123 =======================================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    
    
    X_train, Y_train = DS12_folds[i][0],DS12_folds[i][3]  
    X_val,Y_val = DS12_folds[i][1],DS12_folds[i][4]
    X_test,Y_test = DS12_folds[i][2],DS12_folds[i][5]
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    
    dataset_train = Dataset_seq_embeddings(X_train,X_train_seq,Y_train)
    dataset_val = Dataset_seq_embeddings(X_val,X_val_seq,Y_val)
    dataset_test = Dataset_seq_embeddings(X_test,X_test_seq,Y_test)
    

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective_DS123(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=False,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS123_fold{0}.pt'.format(i+1)),
                                              X_length = 1280,
                                              max_pool_kernel_size = 4),
                   
                   n_trials = N_TRIALS,
                  n_jobs=-1)
    
    
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS123_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS123_fold{0}.pt'.format(i)),'rb'))
    # Build model using best hyperparameters:
    model = model_DS123_build(best_params,input_dim_desc = DS12_folds[0][0].shape[1], X_length = 1280,stride_CNN = 1,
                 conv_dilation1 = 1,padding1 = 0, max_pool_kernel_size = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS123_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    #print("Re-training best model")
    Vivo_train_results = train_and_validate_1CNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path =  os.path.join(ROOT_DIR, 'models','VIVO_FFNN_DS123_fold{0}.pt'.format(i+1)))
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_1CNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS123_fold{0}.pt'.format(i+1)))
    
    CV_folds_test_vivo[i] = Vivo_test_results
with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS123.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )


# Loop over test folds DS13 =======================================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS12_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    
    
    X_train, Y_train = DS1_folds[i][0],DS1_folds[i][3]  
    X_val,Y_val = DS1_folds[i][1],DS1_folds[i][4]
    X_test,Y_test = DS1_folds[i][2],DS1_folds[i][5]
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    
    dataset_train = Dataset_seq_embeddings(X_train,X_train_seq,Y_train)
    dataset_val = Dataset_seq_embeddings(X_val,X_val_seq,Y_val)
    dataset_test = Dataset_seq_embeddings(X_test,X_test_seq,Y_test)
    

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective_DS123(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=False,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS13_fold{0}.pt'.format(i+1)),
                                              X_length = 1280,
                                              max_pool_kernel_size = 4),
                   n_jobs=-1,
                   n_trials = N_TRIALS)
    
    
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS13_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS13_fold{0}.pt'.format(i)),'rb'))
    # Build model using best hyperparameters:
    model = model_DS123_build(best_params,input_dim_desc = DS1_folds[0][0].shape[1], X_length = 1280,stride_CNN = 1,
                 conv_dilation1 = 1,padding1 = 0, max_pool_kernel_size = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS123_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    #print("Re-training best model")
    Vivo_train_results = train_and_validate_1CNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path =  os.path.join(ROOT_DIR, 'models','VIVO_FFNN_DS13_fold{0}.pt'.format(i+1)))
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_1CNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS13_fold{0}.pt'.format(i+1)))
    
    CV_folds_test_vivo[i] = Vivo_test_results
with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS13.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )


# Loop over test folds DS23 =======================================================================================================================
CV_folds_test_vivo = {}
for i in range(len(DS2_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    
    
    X_train, Y_train = DS2_folds[i][0],DS2_folds[i][3]  
    X_val,Y_val = DS2_folds[i][1],DS2_folds[i][4]
    X_test,Y_test = DS2_folds[i][2],DS2_folds[i][5]
    scaler_Y = pickle.load(open(os.path.join(ROOT_DIR, 'data','processed', 'Scaler_Y_' + '{0}.pkl'.format(i)),'rb'))
    
    X_train_seq = DS3_folds[i][0].astype(np.float64)
    X_train_seq.set_index(X_train.index,inplace = True)
    X_val_seq = DS3_folds[i][1].astype(np.float64)
    X_val_seq.set_index(X_val.index,inplace = True)
    X_test_seq = DS3_folds[i][2].astype(np.float64)
    X_test_seq.set_index(X_test.index,inplace = True)
    
    
    dataset_train = Dataset_seq_embeddings(X_train,X_train_seq,Y_train)
    dataset_val = Dataset_seq_embeddings(X_val,X_val_seq,Y_val)
    dataset_test = Dataset_seq_embeddings(X_test,X_test_seq,Y_test)
    

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective_DS123(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=False,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS23_fold{0}.pt'.format(i+1)),
                                              X_length = 1280,
                                              max_pool_kernel_size = 4),
                   n_jobs=-1,
                   n_trials = N_TRIALS)
    
    
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS23_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS23_fold{0}.pt'.format(i)),'rb'))
    # Build model using best hyperparameters:
    model = model_DS123_build(best_params,input_dim_desc = DS2_folds[0][0].shape[1], X_length = 1280,stride_CNN = 1,
                 conv_dilation1 = 1,padding1 = 0, max_pool_kernel_size = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS123_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    #print("Re-training best model")
    Vivo_train_results = train_and_validate_1CNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path =  os.path.join(ROOT_DIR, 'models','VIVO_FFNN_DS23_fold{0}.pt'.format(i+1)))
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_1CNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS23_fold{0}.pt'.format(i+1)))
    
    CV_folds_test_vivo[i] = Vivo_test_results
with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS23.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )

# Loop over test folds DS3 =======================================================================================================================

CV_folds_test_vivo = {}
for i in range(len(DS2_folds)): 
    print("=================BEGINNING FOLD {0} ==============".format(i+1))
    
    
    X_train, Y_train = DS2_folds[i][0],DS2_folds[i][3]  
    X_val,Y_val = DS2_folds[i][1],DS2_folds[i][4]
    X_test,Y_test = DS2_folds[i][2],DS2_folds[i][5]
    
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
    
    
    dataset_train = Dataset_seq_embeddings(X_train,X_train_seq,Y_train)
    dataset_val = Dataset_seq_embeddings(X_val,X_val_seq,Y_val)
    dataset_test = Dataset_seq_embeddings(X_test,X_test_seq,Y_test)
    

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective_DS123(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
                                              Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=False,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS3_fold{0}.pt'.format(i+1)),
                                              X_length = 1280,
                                              max_pool_kernel_size = 4),
                   n_jobs=-1,
                   n_trials = N_TRIALS)
    
    
    trial_ = study.best_trial
    with open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS3_fold{0}.pt'.format(i)),'wb') as f:pickle.dump(trial_.params,f )
    print("Best hyperparameters for fold {0} is saved".format(i+1))
    print(f" best parameters for this fold {trial_.params}")
    
    best_params = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'Optuna_DS3_fold{0}.pt'.format(i)),'rb'))
    # Build model using best hyperparameters:
    model = model_DS123_build(best_params,input_dim_desc = DS2_folds[0][0].shape[1], X_length = 1280,stride_CNN = 1,
                 conv_dilation1 = 1,padding1 = 0, max_pool_kernel_size = 4)
    #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS123_fold{0}.pt".format(i+1)))
    
    # Train model (again) on best hyperparameters for train/val diagnostic plots:
    #print("Re-training best model")
    Vivo_train_results = train_and_validate_1CNN(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                            scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path =  os.path.join(ROOT_DIR, 'models','VIVO_FFNN_DS3_fold{0}.pt'.format(i+1)))
    
    # Make test set into dataloader:
    test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)
    
    # Use best model to evaluate on unseen test data:
    Vivo_test_results = test_1CNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path =  os.path.join(ROOT_DIR, 'models', 'VIVO_FFNN_DS3_fold{0}.pt'.format(i+1)))
    
    CV_folds_test_vivo[i] = Vivo_test_results
with open(os.path.join(ROOT_DIR, 'data','processed','ANN_outer_5_test_DS3.pkl'),'wb') as f:pickle.dump(CV_folds_test_vivo,f )





