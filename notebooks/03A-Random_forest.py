
'''
Random forest models on every descriptorset combination (15 in total). 
Sklean pipeline is used to ease the scaling and PCA pre-processeing before using Random Forest regressor for prediction output.
Output: 
- Dict of outer test errors for each descriptorset combination. 
- Dict of dataframes with predictions/ observations for each PK parameter in long format
- Dict of hyperparameters for each descriptorset for each fold

The script also contain SHAP analysis for RF

'''
from matplotlib import pyplot as plt
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pickle
import os
import random
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse, make_scorer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from itertools import repeat, chain
import seaborn as sns
sys.path.append('../src/insulin_pk/utils/') 
from utils import *
import joblib

# Set seed
seed = 1
np.random.seed(seed)
random.seed(seed)

get_ipython().system('jupyter nbconvert --to script "03A-Random_forest.ipynb"')



## Load ECFP4 descriptors (For Figure S16)
res2 = joblib.load("../data/raw/ecfp4.joblib")
res2.set_index("nncno",inplace=True)
res2 = res2.dropna()
res3 = res2.ecfp4_feature.apply(pd.Series)
res3 = res3.add_prefix('DS5_')


## Load data from output of "get_data.ipynb"
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
# Set overall parameters for optimization:
KFold_outer = GroupKFold(n_splits=5)
param_grid = {
    'RF__n_estimators': [200],
    'RF__min_samples_leaf': [1,3,5,7,9],
    'RF__max_features': ["sqrt","log2"]
    } 
Full_data = Full_data[Full_data.index.isin(res3.index)]
Full_data = pd.concat([Full_data,res3],axis=1) 


# # use RF with DS5


def contruct_descriptor_sets(Full_dataset,DS_set,with_PK=False):
    '''
    Function to create the correct descriptorsets according to the input names
    input: Full dataset and string indicating the descriptorset to include
    output: Reduced dataset (in columns) for the individual descriptorset
    '''
   
    
    DS1 = Full_dataset[Full_dataset.columns[pd.Series(Full_dataset.columns).str.startswith('DS1_')]]
    DS2 = Full_dataset[Full_dataset.columns[pd.Series(Full_dataset.columns).str.startswith('DS2_')]]
    DS3 = Full_dataset[Full_dataset.columns[pd.Series(Full_dataset.columns).str.startswith('DS3_')]]
    DS4 = Full_dataset[Full_dataset.columns[pd.Series(Full_dataset.columns).str.startswith('DS4_')]]
    DS5 = Full_dataset[Full_dataset.columns[pd.Series(Full_dataset.columns).str.startswith('DS5_')]]
    
    
    data_final = pd.DataFrame()
    if DS_set.str.contains("1").values[0]:
        data_final = pd.concat([data_final,DS1],axis=1)
    if DS_set.str.contains("2").values[0]:
        data_final = pd.concat([data_final,DS2],axis=1)
    if DS_set.str.contains("3").values[0]:
        data_final = pd.concat([data_final,DS3],axis=1)  
    if DS_set.str.contains("4").values[0]:
        data_final = pd.concat([data_final,DS4],axis=1)  
    if DS_set.str.contains("5").values[0]:
        data_final = pd.concat([data_final,DS5],axis=1)  
            
    
    if with_PK:
        PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']
        PK_data = Full_dataset[PK_names]
        data_final = pd.concat([data_final,PK_data],axis=1) 
    
    #data_final = pd.concat([data_final,PK_data],axis=1)    
    return data_final  



# Initialize dicts for outer test error and scatterplot
Output_dict = {}
Output_scatter_dict = {}
Output_Hyperparameter_dict = {}

# Define descriptor combinations
Descriptor_combinations = ["DS5","DS15","DS125"]
# Loop over descriptorsets
for des in Descriptor_combinations:
    
    # Extract data
    data = contruct_descriptor_sets(Full_data,pd.Series(des),with_PK = True)
    print("---- Evaluating descriptorset: {} with {} variables ------".format(des,data.shape[1]-3))
    # Split into response and descriptors
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
        
        
        # Simple scaling of features if you numeric input (DS1,DS2,DS12):
        preprocessor = Pipeline([('Scale',StandardScaler())])
        numeric_features = list(X_train_outer.select_dtypes(include=[np.number]).columns)
        # Add PCA on smiles if we're having those descriptors in the set:
        numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS5_')]
        ec_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS5_')].columns
        ec_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
        preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features),('ec_pipe',ec_pipe,ec_names)],remainder='passthrough')
        
        
        # Define estimator with the relevant preprocessor accordign to descriptorset:
        estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = 100,random_state = 42,min_samples_leaf=5,max_features="sqrt"))])
        # Gridsearch for hyperparameters
        GridObject = GridSearchCV(cv = KFold_inner,estimator = estimator, param_grid = param_grid, scoring = make_scorer(mse,greater_is_better=False), n_jobs = -1, verbose = 1,return_train_score=True,refit=True)
        GridObject.fit(X_train_outer,Y_train_outer)
        # Save hyperparameters
        hyperparameter_folds[j] = GridObject.best_params_
        
        # Use final model to predict y pred on X test. 
        ypred = pd.DataFrame(GridObject.predict(X_test),columns = Y_test.columns,index = Y_test.index)
        # Calculate RMSE only on CL (after review comment)
        test_save.append(np.sqrt(mse(Y_test.iloc[:,0],ypred.iloc[:,0])))
        # Save pandas dataframe for scatterplots
        ypred = ypred.assign(Type = "predictions")
        Y_test = Y_test.assign(Type = "observations")
        scatter_long_tmp = pd.concat([ypred,Y_test],axis=0).reset_index().melt(id_vars = ["nncno","Type"]).assign(Fold = j)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
        
    # Save information for each descriptorset in dicts    
    Output_dict[des]= test_save
    Output_scatter_dict[des] = scatter_long_pd
    Output_Hyperparameter_dict[des] = hyperparameter_folds
   
# Save ouput files
with open('../data/processed/Random_forest_outer_test_scorer_DS5_PCA.pkl','wb') as f:pickle.dump(Output_dict,f )
with open('../data/processed/Random_forest_scatter_file_DS5_PCA.pkl','wb') as f:pickle.dump(Output_scatter_dict,f )
with open('../data/processed/Random_forest_hyperparameter_file_DS5_PCA.pkl','wb') as f:pickle.dump(Output_Hyperparameter_dict,f )
  


# Initialize dicts for outer test error and scatterplot
Output_dict = {}
Output_scatter_dict = {}
Output_Hyperparameter_dict = {}

# Define descriptor combinations
Descriptor_combinations = ["DS5","DS15","DS125"]
# Loop over descriptorsets
for des in Descriptor_combinations:
    
    # Extract data
    data = contruct_descriptor_sets(Full_data,pd.Series(des),with_PK = True)
    print("---- Evaluating descriptorset: {} with {} variables ------".format(des,data.shape[1]-3))
    # Split into response and descriptors
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
        
        
        # Simple scaling of features if you numeric input (DS1,DS2,DS12):
        preprocessor = Pipeline([('Scale',StandardScaler())])
        numeric_features = list(X_train_outer.select_dtypes(include=[np.number]).columns)
        
        preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features)],remainder='passthrough')
        
        
        # Define estimator with the relevant preprocessor accordign to descriptorset:
        estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = 100,random_state = 42,min_samples_leaf=5,max_features="sqrt"))])
        # Gridsearch for hyperparameters
        GridObject = GridSearchCV(cv = KFold_inner,estimator = estimator, param_grid = param_grid, scoring = make_scorer(mse,greater_is_better=False), n_jobs = -1, verbose = 1,return_train_score=True,refit=True)
        GridObject.fit(X_train_outer,Y_train_outer)
        # Save hyperparameters
        hyperparameter_folds[j] = GridObject.best_params_
        
        # Use final model to predict y pred on X test. 
        ypred = pd.DataFrame(GridObject.predict(X_test),columns = Y_test.columns,index = Y_test.index)
        # Calculate RMSE only on CL (after review comment)
        test_save.append(np.sqrt(mse(Y_test.iloc[:,0],ypred.iloc[:,0])))
        # Save pandas dataframe for scatterplots
        ypred = ypred.assign(Type = "predictions")
        Y_test = Y_test.assign(Type = "observations")
        scatter_long_tmp = pd.concat([ypred,Y_test],axis=0).reset_index().melt(id_vars = ["nncno","Type"]).assign(Fold = j)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
        
    # Save information for each descriptorset in dicts    
    Output_dict[des]= test_save
    Output_scatter_dict[des] = scatter_long_pd
    Output_Hyperparameter_dict[des] = hyperparameter_folds
   
# Save ouput files
with open('../data/processed/Random_forest_outer_test_scorer_DS5.pkl','wb') as f:pickle.dump(Output_dict,f )
with open('../data/processed/Random_forest_scatter_file_DS5.pkl','wb') as f:pickle.dump(Output_scatter_dict,f )
with open('../data/processed/Random_forest_hyperparameter_file_DS5.pkl','wb') as f:pickle.dump(Output_Hyperparameter_dict,f )
  

## Load data from output of "get_data.ipynb"
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
# Set overall parameters for optimization:
KFold_outer = GroupKFold(n_splits=5)
param_grid = {
    'RF__n_estimators': [200],
    'RF__min_samples_leaf': [1,3,5,7,9],
    'RF__max_features': ["sqrt","log2"]
    } 

# Initialize dicts for outer test error and scatterplot
Output_dict = {}
Output_dict_PK = {}
Output_scatter_dict = {}

# Define descriptor combinations
Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
# Following division is used for splitting the descriptorsets conveniently when making sklearns pipeline as preprocessor later on: 
Descriptor_with_smiles = pd.Series(["DS14","DS4","DS124","DS24"])
Descriptor_with_esm = pd.Series(["DS13","DS3","DS123","DS23"])
Descriptors_with_both = pd.Series(["DS1234","DS134","DS234","DS34"])

# Loop over descriptorsets
for des in Descriptor_combinations:
    
    # Extract data
    data = contruct_descriptor_sets(Full_data,pd.Series(des),with_PK = True)
    print("---- Evaluating descriptorset: {} with {} variables ------".format(des,data.shape[1]-3))
    # Split into response and descriptors
    X_vivo = data.drop(PK_names,axis=1)
    Y_vivo = data[PK_names]
    Y_vivo = np.log(Y_vivo)
    # Initiate list and dicts for savings
    test_save = []
    pandas_save_predict = {}
    pandas_save_target = {} 
    scatter_long_pd = pd.DataFrame()
    hyperparameter_folds = {}
    RMSE_PK_PD = pd.DataFrame()
    
    
    
    # Loop over datafolds:
    # Note that this splitting is identical to the splitting in "Split_data.ipynb" to ensure fair comparison using paired t-test between RF and ANN later on
    for j, (outer_train_idx, test_idx) in enumerate(KFold_outer.split(X_vivo, groups = X_vivo.index.get_level_values("nncno"))):
        X_train_outer, X_test, Y_train_outer, Y_test = X_vivo.iloc[outer_train_idx], X_vivo.iloc[test_idx], Y_vivo.iloc[outer_train_idx], Y_vivo.iloc[test_idx]
        KFold_inner = GroupShuffleSplit(n_splits=1,test_size = 0.15,random_state=42).split(X_train_outer,groups =X_train_outer.index.get_level_values("nncno"))    
        #KFold_inner = GroupKFold(n_splits=5)
        # Simple scaling of features if you numeric input (DS1,DS2,DS12):
        preprocessor = Pipeline([('Scale',StandardScaler())])
        numeric_features = list(X_train_outer.select_dtypes(include=[np.number]).columns)
        # Add PCA on smiles if we're having those descriptors in the set:
        if pd.Series(des).isin(Descriptor_with_smiles).values[0]:
            numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS4_')]
            smiles_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS4_')].columns
            smiles_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
            preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features),('smilespca_pipe',smiles_pipe,smiles_names)],remainder='passthrough')
         # Add PCA on ESM embedding if we're having those descriptors in the set:
        elif pd.Series(des).isin(Descriptor_with_esm).values[0]:
            numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS3_')]
            esm_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS3_')].columns
            esm_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
            preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features),('esm_pca_pipe',esm_pipe,esm_names)],remainder='passthrough')
        # PCA on both if we have both sequential data in descriptorset:
        elif pd.Series(des).isin(Descriptors_with_both).values[0]:
            numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS4_')]
            numeric_features = [numeric_features for numeric_features in numeric_features if not numeric_features.startswith('DS3_')]
            esm_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS3_')].columns
            smiles_names = X_train_outer.iloc[:,X_train_outer.columns.str.startswith('DS4_')].columns
            esm_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
            smiles_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
            preprocessor = ColumnTransformer(transformers=[('num_pipe', preprocessor, numeric_features),('smiles_pca_pipe',smiles_pipe,smiles_names),
                                                          ('esm_pca_pipe',esm_pipe,esm_names)],remainder='passthrough')
            
    
        # Define estimator with the relevant preprocessor accordign to descriptorset:
        estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = 100,random_state = 42,min_samples_leaf=5,max_features="sqrt"))])
        # Gridsearch for hyperparameters
        GridObject = GridSearchCV(cv = KFold_inner,estimator = estimator, param_grid = param_grid, scoring = make_scorer(mse,greater_is_better=False), n_jobs = -1, verbose = 1,return_train_score=True,refit=True)
        GridObject.fit(X_train_outer,Y_train_outer,groups = X_train_outer.index.get_level_values("nncno"))
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
    Output_dict[des]= test_save
    Output_scatter_dict[des] = scatter_long_pd
    Output_Hyperparameter_dict[des] = hyperparameter_folds
   
# Save ouput files
with open('../data/processed/Random_forest_outer_test_scorer.pkl','wb') as f:pickle.dump(Output_dict,f )
with open('../data/processed/Random_forest_scatter_file.pkl','wb') as f:pickle.dump(Output_scatter_dict,f )
with open('../data/processed/Random_forest_hyperparameter_file.pkl','wb') as f:pickle.dump(Output_Hyperparameter_dict,f )
  


# # SHAP values for Random Forest DS124 (no groups)

## Split the full dataset in 80% Training, 20% Test
Data_124 = contruct_descriptor_sets(Full_data,pd.Series("DS124"),with_PK = True)
X = Data_124.drop(PK_names,axis=1)
Y = Data_124[PK_names]
Y = np.log(Y)
train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 42).split(X, groups=X.index.get_level_values("nncno")))
X_train = X.iloc[train_inds,:]
Y_train = Y.iloc[train_inds,:]
X_test = X.iloc[test_inds,:]
Y_test = Y.iloc[test_inds,:]


def find_best_hyp(DS):
    '''
    Extract best hyperparameter setting
    input:
        - DS: String that indicates which descriptorset
    Output:
        - Dict with best (mode) hyperparameter settings
    '''

    Hyperparameters_pd = d = pd.DataFrame(np.zeros((len(Output_Hyperparameter_dict[DS]), 3)))
    for i in range(len(Output_Hyperparameter_dict["DS124"])):
        Hyperparameters_pd.iloc[i,:] = list(Output_Hyperparameter_dict["DS124"][i].values())
    best_parameters = Hyperparameters_pd.mode()


    param_grid_optimal = {
        'RF__n_estimators': int(best_parameters[2].item()),
        'RF__min_samples_leaf': int(best_parameters[1].item()),
        'RF__max_features': best_parameters[0].item()
        } 
    return param_grid_optimal


# In[10]:


best_hyperparameters = find_best_hyp("DS124")


# In[11]:


numeric_features_names = np.array(X.loc[:, X.columns.str.startswith("DS1") | X.columns.str.startswith("DS2")].columns)
smiles_names = np.array(X.iloc[:,X.columns.str.startswith('DS4_')].columns)
            
num_pipe = Pipeline([('Scale',StandardScaler())])
smiles_pipe = smiles_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
preprocessor = ColumnTransformer(transformers=[('num_pipe', num_pipe, numeric_features_names),('smiles_pca_pipe',smiles_pipe,smiles_names)],remainder='drop')
estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = best_hyperparameters["RF__n_estimators"],
                                                                              random_state = 42,min_samples_leaf=best_hyperparameters["RF__min_samples_leaf"]
                                                                              ,max_features=best_hyperparameters["RF__max_features"]))])
        


# In[12]:


preprocessed_X_test = pd.DataFrame(preprocessor.fit_transform(X_test))
preprocessed_X_test.columns = np.concatenate((numeric_features_names,smiles_names[0:20]))
preprocessed_X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
preprocessed_X_train.columns = np.concatenate((numeric_features_names,smiles_names[0:20]))

preprocessed_X_full  = pd.concat([preprocessed_X_train,preprocessed_X_test],axis=0)

RF_object = estimator.fit(X_train,Y_train)
explainer = shap.TreeExplainer(RF_object['RF'])
shap_values = explainer.shap_values(preprocessed_X_full)
with open('../data/processed/Random_forest_SHAP_values_full.pkl','wb') as f:pickle.dump(shap_values,f )
with open('../data/processed/Random_forest_SHAP_values_X_test_full.pkl','wb') as f:pickle.dump(preprocessed_X_test,f )
with open('../data/processed/Random_forest_SHAP_values_Y_train_full.pkl','wb') as f:pickle.dump(Y_train,f )


# # SHAP Random Forest (Groups)

# In[13]:


X_corr = pd.DataFrame(preprocessed_X_train,columns=preprocessed_X_train.columns).corr()
plt.figure(figsize=(3,15))
dissimilarity = 1 - abs(X_corr)
Z = linkage(squareform(dissimilarity), 'complete')
#Z = linkage(X_corr, 'complete')

dendrogram(Z, labels=preprocessed_X_train.columns, orientation='right', 
 leaf_rotation=0,color_threshold = 0.1);
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=10)
ax.tick_params(axis='y', which='major', labelsize=8)
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_Dendogram.png",bbox_inches = 'tight')


# In[14]:


labels = fcluster(Z, 0.1, criterion='distance')
#plt.savefig("/home/kyei/Project1/Figures/Dendogram_DS1234_0_2.png")
Labels_pd = pd.DataFrame(pd.Series(preprocessed_X_train.columns),columns=["Features"])
Labels_pd["Cluster"] = labels
Dict_cluster = {k: g["Features"].tolist() for k,g in Labels_pd.groupby("Cluster")}



revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))


def grouped_shap(shap_vals, features, groups):
    '''
    Function to sum up shapvalues within highly correlated groups (reference: Scott Lundberg https://medium.com/@scottmlundberg/good-question-6229a343819f)
    '''
    
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    
    return shap_grouped


pandas_cluster = pd.DataFrame(pd.Series(Dict_cluster, name='Features'))
pandas_cluster['liststring'] = pandas_cluster['Features'].apply(lambda x: ','.join(map(str, x)))
pandas_cluster['liststring'] = pandas_cluster['liststring'].str.slice(0,150)


# In[15]:


groupmap = revert_dict(Dict_cluster)
shap_Tdf = pd.DataFrame(preprocessed_X_full, columns=pd.Index(preprocessed_X_full.columns, name='features')).T
shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
pd_features_grouped = grouped_shap(preprocessed_X_full,preprocessed_X_full.columns,Dict_cluster)
pd_features_grouped.columns = pandas_cluster.liststring


# In[ ]:


new_col_names = np.array(pd_features_grouped.columns)
new_col_names[np.where(new_col_names == "DS2_molwt,DS2_tpsa,DS4_0")[0].item()] = "DS2_molwt,DS2_tpsa,DS4_PC1"
new_col_names[np.where(new_col_names == "DS4_3")[0].item()] = "DS4_PC3"
new_col_names[np.where(new_col_names == "DS4_8")[0].item()] = "DS4_PC8"
new_col_names[np.where(new_col_names == "DS4_13")[0].item()] = "DS4_PC13"
new_col_names[np.where(new_col_names == "DS4_2")[0].item()] = "DS4_PC2"
new_col_names[np.where(new_col_names == "DS4_7")[0].item()] = "DS4_PC7"
new_col_names[np.where(new_col_names == "DS1_PEP_BOND_COUNT,DS1_length,DS1_mw")[0].item()] = "DS1_pep_bond_count,DS1_length,DS1_mw"



# In[42]:


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
    shap_type = grouped_shap(shap_values[i-1], preprocessed_X_test.columns,Dict_cluster)
    shap_type.columns = pd_cluster["liststring"]
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    
    #f = plt.figure()
    newcmp = sns.color_palette("crest_r", as_cmap=True)
    shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = new_col_names,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (10,10),color_bar_label='Feature value (sum)')
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
plt.savefig("../reports/figures/Figure_S5_DS124_shap.png",bbox_inches = 'tight',dpi=500)


# In[43]:


columns = 1
rows = 3
what_PK = 0
names = ["CL","T1-2","MRT"]
#plt.rcParams.update({'font.size': 10})
label_size = 13
pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
shap_type = grouped_shap(shap_values[0], preprocessed_X_test.columns,Dict_cluster)
shap_type.columns = pd_cluster["liststring"]
newcmp = sns.color_palette("crest_r", as_cmap=True)

fig = plt.figure(figsize=(14, 4))
newcmp = sns.color_palette("crest_r", as_cmap=True)
shap.summary_plot(shap_type.values, pd_features_grouped,feature_names = new_col_names,class_names =Y_train.columns ,max_display = 10,show=False,cmap=newcmp, plot_size = (14,4),color_bar_label='Feature value (sum)')
#plt.savefig("/home/kyei/Project1/Figures/RF_DS124_shap_total.png",bbox_inches = 'tight')
plt.gca().set_xlim(-0.6, 0.6)
plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.xlabel('xlabel', fontsize=label_size)
#plt.figsize=(18, 40)
plt.title("{}".format(names[0]),fontsize=label_size)
plt.gca().set_xlabel('')
plt.savefig("../reports/figures/Figure6a.png",bbox_inches = 'tight',dpi = 500)


# In[13]:


### Plot the two first PCs with number of protractors as colors. 
#preprocessed_X_train_copy_for_plot = preprocessed_X_train.copy()
#preprocessed_X_train_copy_for_plot["DS2_num"] = np.round(preprocessed_X_train_copy_for_plot["DS2_num"],2)
#preprocessed_X_train_copy_for_plot["DS2_num"] = np.where(preprocessed_X_train_copy_for_plot["DS2_num"] == -1.42,0,
#                                                         np.where(preprocessed_X_train_copy_for_plot["DS2_num"] == 0.29,1,
#                                                                  np.where(preprocessed_X_train_copy_for_plot["DS2_num"] == 2,2,
#                                                                           np.where(preprocessed_X_train_copy_for_plot["DS2_num"] == 3.71,3,"error"))))
#sns.scatterplot(data = preprocessed_X_train_copy_for_plot,x = "DS4_0", y = "DS4_1",hue = "DS2_num")


# In[29]:


SHAP_features_repeats_CL = {}
SHAP_features_repeats_T12 = {}
SHAP_features_repeats_MRT = {}

SHAP_features_repeats_CL = pd.DataFrame()
SHAP_features_repeats_T12 = pd.DataFrame()
SHAP_features_repeats_MRT = pd.DataFrame()

for jj in range(5):

    ## Split the full dataset in 80% Training, 20% Test
    Data_124 = contruct_descriptor_sets(Full_data,pd.Series("DS124"),with_PK = True)
    X = Data_124.drop(PK_names,axis=1)
    Y = Data_124[PK_names]
    Y = np.log(Y)
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = seeds[jj]).split(X, groups=X.index.get_level_values("nncno")))
    X_train = X.iloc[train_inds,:]
    Y_train = Y.iloc[train_inds,:]
    X_test = X.iloc[test_inds,:]
    Y_test = Y.iloc[test_inds,:]

    best_hyperparameters = find_best_hyp("DS124")

    numeric_features_names = np.array(X.loc[:, X.columns.str.startswith("DS1") | X.columns.str.startswith("DS2")].columns)
    smiles_names = np.array(X.iloc[:,X.columns.str.startswith('DS4_')].columns)

    num_pipe = Pipeline([('Scale',StandardScaler())])
    smiles_pipe = smiles_pipe = Pipeline([('Scale',StandardScaler()), ('PCA', PCA(n_components = 20))])
    preprocessor = ColumnTransformer(transformers=[('num_pipe', num_pipe, numeric_features_names),('smiles_pca_pipe',smiles_pipe,smiles_names)],remainder='drop')
    estimator = Pipeline([('preproc', preprocessor), ('RF', RandomForestRegressor(criterion='mse', n_estimators = best_hyperparameters["RF__n_estimators"],
                                                                              random_state = 42,min_samples_leaf=best_hyperparameters["RF__min_samples_leaf"]
                                                                              ,max_features=best_hyperparameters["RF__max_features"]))])

    preprocessed_X_test = pd.DataFrame(preprocessor.fit_transform(X_test))
    preprocessed_X_test.columns = np.concatenate((numeric_features_names,smiles_names[0:20]))
    preprocessed_X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
    preprocessed_X_train.columns = np.concatenate((numeric_features_names,smiles_names[0:20]))

    RF_object = estimator.fit(X_train,Y_train)
    explainer = shap.TreeExplainer(RF_object['RF'])
    shap_values = explainer.shap_values(preprocessed_X_test)



    X_corr = pd.DataFrame(preprocessed_X_train,columns=preprocessed_X_train.columns).corr()
    #plt.figure(figsize=(3,15))
    dissimilarity = 1 - abs(X_corr)
    Z = linkage(squareform(dissimilarity), 'complete')
    #Z = linkage(X_corr, 'complete')

    #dendrogram(Z, labels=preprocessed_X_train.columns, orientation='right', 
    # leaf_rotation=0,color_threshold = 0.1);
    #ax = plt.gca()
    #ax.tick_params(axis='x', which='major', labelsize=10)
    #ax.tick_params(axis='y', which='major', labelsize=8)
    #plt.savefig("/home/kyei/Project1/Figures/RF_DS124_Dendogram.png",bbox_inches = 'tight')



    labels = fcluster(Z, 0.1, criterion='distance')
    #plt.savefig("/home/kyei/Project1/Figures/Dendogram_DS1234_0_2.png")
    Labels_pd = pd.DataFrame(pd.Series(preprocessed_X_train.columns),columns=["Features"])
    Labels_pd["Cluster"] = labels
    Dict_cluster = {k: g["Features"].tolist() for k,g in Labels_pd.groupby("Cluster")}



    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

    pandas_cluster = pd.DataFrame(pd.Series(Dict_cluster, name='Features'))
    pandas_cluster['liststring'] = pandas_cluster['Features'].apply(lambda x: ','.join(map(str, x)))
    pandas_cluster['liststring'] = pandas_cluster['liststring'].str.slice(0,150)


    groupmap = revert_dict(Dict_cluster)
    shap_Tdf = pd.DataFrame(preprocessed_X_test, columns=pd.Index(preprocessed_X_test.columns, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    pd_features_grouped = grouped_shap(preprocessed_X_test,preprocessed_X_test.columns,Dict_cluster)
    pd_features_grouped.columns = pandas_cluster.liststring
    
    
    shap_PK_save = {}

    for i in range(3):
        pd_cluster = pd.DataFrame(pandas_cluster['liststring'])
        shap_grouped = grouped_shap(shap_values[i], preprocessed_X_test.columns,Dict_cluster)
        shap_grouped.columns = pd_cluster["liststring"]
        shap_PK_save[i] = shap_grouped
    
    top10_shap_features_CL = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[0]).mean(0))]).iloc[::-1][:10]
    top10_shap_features_T12 = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[1]).mean(0))]).iloc[::-1][:10]
    top10_shap_features_MRT = pd.DataFrame(pd_features_grouped.columns[np.argsort(np.abs(shap_PK_save[2]).mean(0))]).iloc[::-1][:10]
        
    SHAP_features_repeats_CL = pd.concat([SHAP_features_repeats_CL,top10_shap_features_CL],axis=0)
    SHAP_features_repeats_T12 = pd.concat([SHAP_features_repeats_T12,top10_shap_features_T12],axis=0)
    SHAP_features_repeats_MRT = pd.concat([SHAP_features_repeats_MRT,top10_shap_features_MRT],axis=0)
    
    

