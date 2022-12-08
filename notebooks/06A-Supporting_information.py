#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse, make_scorer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from itertools import repeat, chain
from torch.utils.data import Dataset, DataLoader
plt.style.use('ggplot')
sns.set(font_scale=2)

import sys
sys.path.append('../src/insulin_pk/utils/') 
from utils import *
Data_with_groups = pd.read_excel("../data/raw/Data_with_groups.xlsx")
Data_with_groups.rename(columns={"NNCNo": "nncno"})
Data_with_groups.set_index("NNCNo",inplace=True)
Data_with_groups = Data_with_groups[~Data_with_groups.index.isin(["0148-0000-1247"])]
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
data_wg = pd.merge(Full_data, Data_with_groups["Groups"], left_index=True, right_index=True)
data_wg.index.name = "nncno"
get_ipython().system('jupyter nbconvert --to script "06A-Supporting_information.ipynb"')


# %%


# Set overall parameters for optimization:
KFold_outer = GroupKFold(n_splits=5)
param_grid = {
    'RF__n_estimators': [200],
    'RF__min_samples_leaf': [1,3,5,7,9],
    'RF__max_features': ["sqrt","log2"]
    } 
def r2_rmse(g):
    '''
    Function to calculate root-mean-square-error (RMSE) between two columns "observations" and "predictions"
    
    '''
    r2 = r2_score( g['observations'], g['predictions'] )
    rmse = np.sqrt( mean_squared_error( g['observations'], g['predictions'] ) )
    return pd.Series(dict(rmse = rmse))


# %%


# Initialize dicts for outer test error and scatterplot
Output_dict = {}
Output_scatter_dict = {}
Output_Hyperparameter_dict = {}

# Define descriptor combinations
groups = ['Other', 'Acylation', 'No attachments', 'Concatenated proteins',
       'Antibody attachment (Fc)', 'AA extensions']

# Loop over descriptorsets
for group in groups:
    
    # Extract data
    data_wg = data_wg.iloc[:,~data_wg.columns.str.startswith('DS3_')]
    # Split into response and descriptors
    data = data_wg[data_wg.Groups == group]
    print("---- Evaluating group: {} with {} variables and {} rows ------".format(group,data.shape[1]-4,data.shape[0]))
    data.drop("Groups",axis=1,inplace=True)
    X_vivo = data.drop(PK_names,axis=1)
    Y_vivo = data[PK_names]
    Y_vivo = np.log(Y_vivo)
    # Initiate list and dicts for savings
    test_save = []
    pandas_save_predict = {}
    pandas_save_target = {} 
    scatter_long_pd = pd.DataFrame()
    hyperparameter_folds = {}
    
    # Loop over datafolds:
    # Note that this splitting is identical to the splitting in "Split_data.ipynb" to ensure fair comparison using paired t-test between RF and ANN later on
    for j, (outer_train_idx, test_idx) in enumerate(KFold_outer.split(X_vivo, groups = X_vivo.index.get_level_values("nncno"))):
        X_train_outer, X_test, Y_train_outer, Y_test = X_vivo.iloc[outer_train_idx], X_vivo.iloc[test_idx], Y_vivo.iloc[outer_train_idx], Y_vivo.iloc[test_idx]
        KFold_inner = GroupShuffleSplit(n_splits=1,test_size = 0.15,random_state=42).split(X_train_outer,groups =X_train_outer.index.get_level_values("nncno"))    
        
        # Simple scaling of features if you numeric input (DS1,DS2,DS12):
        preprocessor = Pipeline([('Scale',StandardScaler())])
        numeric_features = list(X_train_outer.select_dtypes(include=[np.number]).columns)
        
        # Add PCA on smiles if we're having those descriptors in the set:
        if data.shape[0] > 20:
            numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS4_')]
            smiles_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS4_')].columns
            smiles_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
            preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features),('smilespca_pipe',smiles_pipe,smiles_names)],remainder='passthrough')
            
        # Define estimator with the relevant preprocessor accordign to descriptorset:
        estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = 100,random_state = 42,min_samples_leaf=5,max_features="sqrt"))])
        # Gridsearch for hyperparameters
        GridObject = GridSearchCV(cv = KFold_inner,estimator = estimator, param_grid = param_grid, scoring = make_scorer(mse,greater_is_better=False), n_jobs = -1, verbose = 1,return_train_score=True,refit=True)
        GridObject.fit(X_train_outer,Y_train_outer)
        # Save hyperparameters
        hyperparameter_folds[j] = GridObject.best_params_
        
        # Use final model to predict y pred on X test. 
        ypred = pd.DataFrame(GridObject.predict(X_test),columns = Y_test.columns,index = Y_test.index)
        # Calculate RMSE as mean across all 3 PK parameters
        test_save.append(np.sqrt(mse(Y_test,ypred)))
        # Save pandas dataframe for scatterplots
        ypred = ypred.assign(Type = "predictions")
        Y_test = Y_test.assign(Type = "observations")
        scatter_long_tmp = pd.concat([ypred,Y_test],axis=0).reset_index().melt(id_vars = ["nncno","Type"]).assign(Fold = j)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
        
    # Save information for each descriptorset in dicts    
    Output_dict[group]= test_save
    Output_scatter_dict[group] = scatter_long_pd
    Output_Hyperparameter_dict[group] = hyperparameter_folds

    
# Save ouput files
with open('../data/processed/Random_forest_outer_test_scorer_DS124_groups.pkl','wb') as f:pickle.dump(Output_dict,f )
with open('../data/processed/Random_forest_scatter_file_DS124_groups.pkl','wb') as f:pickle.dump(Output_scatter_dict,f )
with open('../data/processed/Random_forest_hyperparameter_file_DS124_groups.pkl','wb') as f:pickle.dump(Output_Hyperparameter_dict,f )
  


# # Groups on ANN results

# %%


optuna_hyp = {}
for i in range(5):
    tmp = pickle.load(open('../models/Optuna_DS1234_fold{0}.pt'.format(i),'rb'))
    optuna_hyp[i] = tmp
#pd.DataFrame(optuna_hyp)
best_median_params = pd.DataFrame(optuna_hyp).median(1).to_dict()
# Dumb way to change the values to int..
best_median_params["Batch_Size"] = int(2)
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
#best_median_params


# %%


print("DS1234 ANN already trained on all groups")

groups = ["Other", 'Acylation', 'No attachments', 'Concatenated proteins',
       'Antibody attachment (Fc)', 'AA extensions']  
PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']
EPOCH = 200
N_TRIALS = 30
# First load the data_folds
for kk in groups:
    
    ## Load data in folds and select only relevant descriptorset:
    DS12_folds = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))
    DS1_folds = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))
    DS2_folds = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))
    DS4_folds = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))
    DS3_folds = pickle.load(open('../data/processed/Data_folds_{}.pkl'.format(kk),'rb'))

    for i in range(len(DS4_folds)):
        for j in range(3):
            DS12_folds[i][j] = contruct_descriptor_sets(DS12_folds[i][j],pd.Series("12"))
            DS1_folds[i][j] = contruct_descriptor_sets(DS1_folds[i][j],pd.Series("1"))
            DS2_folds[i][j] = contruct_descriptor_sets(DS2_folds[i][j],pd.Series("2"))
            DS3_folds[i][j] = contruct_descriptor_sets(DS3_folds[i][j],pd.Series("3"))
            DS4_folds[i][j] = contruct_descriptor_sets(DS4_folds[i][j],pd.Series("4"))
    print("calculations for {}, with {} data points".format(kk,DS12_folds[0][0].shape[0] + DS12_folds[0][1].shape[0] + DS12_folds[0][2].shape[0]))    
    # Loop over test folds DS1234
    CV_folds_test_vivo = {}
    for i in range(len(DS12_folds)): 
        print("=================BEGINNING FOLD {0} ==============".format(i+1))
        X_train, Y_train = DS12_folds[i][0].astype(np.float64),DS12_folds[i][3].astype(np.float32)  
        X_val,Y_val = DS12_folds[i][1].astype(np.float64),DS12_folds[i][4].astype(np.float32)
        X_test,Y_test = DS12_folds[i][2].astype(np.float64),DS12_folds[i][5].astype(np.float32)
        scaler_Y = pickle.load(open('../data/processed/Scaler_Y_{}_{}.pkl'.format(kk,i),'rb'))

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

        
        ## Use fixed batch size of 2 to be able to handle (very) small data sets:
        #study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
        #study.optimize(lambda trial:objective_DS1234_fixed_batch_size(trial,Data_train = dataset_train ,Data_Val = dataset_val ,
        #                                         Scaler_Y= scaler_Y,EPOCH=EPOCH,save_model=True,save_path = "../models/VIVO_FFNN_DS1234_fold{}_{}.pt".format(i+1,kk),
        #                                          input_dim_desc = X_train.shape[1]),
        #                                          n_trials = N_TRIALS)
        #trial_ = study.best_trial
        #with open('../models/Optuna_DS1234_fold{}_{}.pkl'.format(i,kk),'wb') as f:pickle.dump(trial_.params,f )
        #print("Best hyperparameters for fold {0} is saved".format(i+1))
        #print(f" best parameters for this fold {trial_.params}")
        #best_params = pickle.load(open('../models/Optuna_DS1234_fold{}_{}.pkl'.format(i,kk),'rb'))
        best_params = best_median_params
        
        # Build model using best hyperparameters:

        model = model_DS1234_build(best_params,input_dim_desc = X_train.shape[1], X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                     conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 4)
        #model.load_state_dict(torch.load("/home/kyei/Project1/Model files/VIVO_FFNN_DS1234_fold{0}.pt".format(i+1)))

        # Train model (again) on best hyperparameters for train/val diagnostic plots:
        Vivo_train_results = train_and_validate(Params = best_params,Model = model,Data_train = dataset_train ,Data_Val = dataset_val,
                                                scaler_Y = scaler_Y,EPOCH = EPOCH,save_model = True,save_path = "../models/VIVO_FFNN_DS1234_fold{}_{}.pt".format(i,kk),
                                               trial = "None")

        # Make test set into dataloader:
        test_loader = DataLoader(dataset = dataset_test,batch_size=X_test.shape[0],shuffle=False,drop_last = True)

        # Use best model to evaluate on unseen test data:
        Vivo_test_results = test_FFNN(model,best_params, test_loader,scaler_Y = scaler_Y,save_path = "../models/VIVO_FFNN_DS1234_fold{}_{}.pt".format(i,kk),Y_data_for_index = Y_test)

        # Save each individual test output for each fold:
        with open("../models/Test_output_DS1234_fold_{}_{}.pkl".format(i,kk),'wb') as f:pickle.dump(Vivo_test_results,f )

        CV_folds_test_vivo[i] = Vivo_test_results

        model.apply(reset_weights)

    
    with open("../data/processed/ANN_outer_5_test_DS1234_{}.pkl".format(kk),'wb') as f:pickle.dump(CV_folds_test_vivo,f )

    print("Completely done with group {}".format(kk))


# %%





# %%




