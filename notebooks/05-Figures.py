

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
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

from scipy.stats import rankdata


sns.set(font_scale=2)
get_ipython().system('jupyter nbconvert --to script "05-Figures.ipynb"')



def r2_rmse(g):
    '''
    Function to calculate root-mean-square-error (RMSE) between two columns "observations" and "predictions"
    
    '''
    r2 = r2_score( g['observations'], g['predictions'] )
    rmse = np.sqrt( mean_squared_error( g['observations'], g['predictions'] ) )
    return pd.Series(dict(rmse = rmse))



def r2_rmse_updated(g):
    '''
    Function to calculate root-mean-square-error (RMSE) between two columns "observations" and "predictions"
    
    '''
    r2 = r2_score( g['observations'], g['predictions'] )
    rmse = np.sqrt( mean_squared_error( g['observations'], g['predictions'] ) )
    return pd.Series(dict(rmse = rmse,r2 = r2))



def rmse_2(tmp):
    rmse = np.sqrt( mean_squared_error( tmp['observations'], tmp['predictions'] ) )
    return rmse
def r2_2(tmp):
    r2 = r2_score( tmp['observations'], tmp['predictions'] )
    return r2


def extract_RMSE_R2_RF(Random_forest_scatter):
    Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
    
    RMSE_total = pd.DataFrame()
    R2_total = pd.DataFrame()
    
    for des in Descriptor_combinations:
        tmp = Random_forest_scatter[des].pivot(index = ["nncno","variable","Fold"],columns = "Type",values = "value").reset_index().set_index("nncno")
        RMSE_table = tmp.groupby(["Fold","variable"]).apply(rmse_2).reset_index()
        RMSE_table.columns = ['Fold', 'variable', 'value']
        RMSE_table = RMSE_table.pivot(index = "Fold",columns = "variable",values = "value")
        RMSE_table["DS"] = des

        R2_table = tmp.groupby(["Fold","variable"]).apply(r2_2).reset_index()
        R2_table.columns = ['Fold', 'variable', 'value']
        R2_table = R2_table.pivot(index = "Fold",columns = "variable",values = "value")
        R2_table["DS"] = des
        
        RMSE_total = pd.concat([RMSE_total,RMSE_table],axis=0)
        R2_total = pd.concat([R2_total,R2_table],axis=0)
        
        
    return RMSE_total, R2_total


def extract_RMSE_R2_RF_original_space(Random_forest_scatter):
    Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
    
    RMSE_total = pd.DataFrame()
    R2_total = pd.DataFrame()
    
    for des in Descriptor_combinations:
        tmp = Random_forest_scatter[des].pivot(index = ["nncno","variable","Fold"],columns = "Type",values = "value").reset_index().set_index("nncno")
        tmp["observations"] = np.exp(tmp["observations"])
        tmp["predictions"] = np.exp(tmp["predictions"])
        
        RMSE_table = tmp.groupby(["Fold","variable"]).apply(rmse_2).reset_index()
        RMSE_table.columns = ['Fold', 'variable', 'value']
        RMSE_table = RMSE_table.pivot(index = "Fold",columns = "variable",values = "value")
        RMSE_table["DS"] = des

        R2_table = tmp.groupby(["Fold","variable"]).apply(r2_2).reset_index()
        R2_table.columns = ['Fold', 'variable', 'value']
        R2_table = R2_table.pivot(index = "Fold",columns = "variable",values = "value")
        R2_table["DS"] = des
        
        RMSE_total = pd.concat([RMSE_total,RMSE_table],axis=0)
        R2_total = pd.concat([R2_total,R2_table],axis=0)
        
        
    return RMSE_total, R2_total




def store_ANN_data():
    '''
    Function to load ANN output data and store all in dictionary
    output:
        - store_dict: Dictionary with Descriptorset x fold x output info
        - outer_rmse_dict: Extracted the outer test scores for each descriptorset for each fold
        
    '''
    
    Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
    
    store_dict = {}
    outer_rmse_dict = {}
    outer_rmse_dict_CL = {}
    outer_rmse_dict_T12 = {}
    outer_rmse_dict_MRT = {}
    
    for i in Descriptor_combinations:
        outer_rmse_list = []
        outer_rmse_list_CL = []
        outer_rmse_list_T12 = []
        outer_rmse_list_MRT = []
        
        data = pickle.load(open('../data/processed/ANN_outer_5_test_{}.pkl'.format(i),'rb'))
        store_dict[i] = data
        for j in range(len(store_dict["DS1"])):
            outer_rmse_list.append(store_dict[i][j]["best_rmse"])
            outer_rmse_list_CL.append(store_dict[i][j]["tmp_assay1"])
            outer_rmse_list_T12.append(store_dict[i][j]["tmp_assay2"])
            outer_rmse_list_MRT.append(store_dict[i][j]["tmp_assay3"])
            
        outer_rmse_dict[i] = outer_rmse_list
        outer_rmse_dict_CL[i] = outer_rmse_list_CL
        outer_rmse_dict_T12[i] = outer_rmse_list_T12
        outer_rmse_dict_MRT[i] = outer_rmse_list_MRT
        
    return store_dict, outer_rmse_dict, outer_rmse_dict_CL, outer_rmse_dict_T12 , outer_rmse_dict_MRT

def store_ANN_data_ts():
    '''
    Function to load ANN output data and store all in dictionary
    output:
        - store_dict: Dictionary with Descriptorset x fold x output info
        - outer_rmse_dict: Extracted the outer test scores for each descriptorset for each fold
        
    '''
    
    Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
    
    store_dict = {}
    outer_rmse_dict = {}
    outer_rmse_dict_CL = {}
    outer_rmse_dict_T12 = {}
    outer_rmse_dict_MRT = {}
    
    for i in Descriptor_combinations:
        outer_rmse_list = []
        outer_rmse_list_CL = []
        outer_rmse_list_T12 = []
        outer_rmse_list_MRT = []
        
        data = pickle.load(open('/home/kyei/Project1_clean/data/processed/ANN_outer_5_test_grouped_{}.pkl'.format(i),'rb'))
        store_dict[i] = data
        for j in range(len(store_dict["DS1"])):
            outer_rmse_list.append(store_dict[i][j]["best_rmse"])
            outer_rmse_list_CL.append(store_dict[i][j]["tmp_assay1"])
            outer_rmse_list_T12.append(store_dict[i][j]["tmp_assay2"])
            outer_rmse_list_MRT.append(store_dict[i][j]["tmp_assay3"])
            
        outer_rmse_dict[i] = outer_rmse_list
        outer_rmse_dict_CL[i] = outer_rmse_list_CL
        outer_rmse_dict_T12[i] = outer_rmse_list_T12
        outer_rmse_dict_MRT[i] = outer_rmse_list_MRT
        
    return store_dict, outer_rmse_dict, outer_rmse_dict_CL, outer_rmse_dict_T12 , outer_rmse_dict_MRT



def paired_t_test_values(data_pd):
    '''
    Calculate paired t_test values
    input:
        - data_pd: Pandas dataframe with outer test results for each descriptor in the columns
    output:
        - t_test_matrix: All combinations of pairs of descriptors and their t test value.
    '''
    
    
    Number_descriptors = data_pd.shape[1]
    t_test_matrix = pd.DataFrame(np.zeros((Number_descriptors, Number_descriptors)))
    for i in range(Number_descriptors):
        for j in range(Number_descriptors):
            t_test_matrix.iloc[i,j] = stats.ttest_rel(data_pd.iloc[:,i], data_pd.iloc[:,j])[1]
    
    t_test_matrix.columns = data_pd.columns
    t_test_matrix.index =data_pd.columns

    return t_test_matrix

def plot_t_test_values(dat_to_plot,title = None):
    '''
    Plot t-test scores for quick overview
    
    '''
    
    mask = np.zeros_like(dat_to_plot)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(15, 10))
        RF_sns = sns.heatmap(round(dat_to_plot,2), cmap='Blues_r', annot=True, 
            annot_kws={"size": 15}, vmin=0, vmax=1,mask=mask)
        ax.set_title(title)

        
def paired_t_test_across_model(RF_data,ANN_data):
    
    Number_descriptors = RF_data.shape[1]
    t_test_vector = pd.Series(np.zeros((Number_descriptors)))
    
    for i in range(Number_descriptors):
            t_test_vector.iloc[i] = stats.ttest_rel(RF_data.iloc[:,i], ANN_data.iloc[:,i])[1]
    
    return t_test_vector
    
        
def calculate_BH(t_test_scores_RF,index):
    '''
    Calculate BH threshold for input T-test matrix. 
    Input:
    
    - t_test_scores_RF: DataFrame og t-test across all tested models (NxN)
    
    - index: Index of which columns (descriptor set) to compare to the rest of the descriptor sets 
    
    output:
    - eval_pval: Matrix of p value, rank and BH value
    - eval_pd: Dataframe of significant descriptorsets
    
    '''
    
    
    t_test_scores_RF_BH = t_test_scores_RF.drop(t_test_scores_RF.iloc[index].name).iloc[:,index]
    t_test_scores_RF_BH_for_index = t_test_scores_RF.drop(t_test_scores_RF.iloc[index].name).iloc[:,index].sort_values().index

    BH_ranks = pd.DataFrame(rankdata(t_test_scores_RF_BH),columns= ["Rank"],index = t_test_scores_RF_BH_for_index)

    BH_values = pd.DataFrame(rankdata(t_test_scores_RF_BH)/7*0.05,columns= ["BH_value"],index = t_test_scores_RF_BH_for_index)
    eval_pval = pd.concat([t_test_scores_RF_BH,BH_ranks,BH_values],axis=1).sort_values(by = "Rank")

    eval_pd = eval_pval[eval_pval[t_test_scores_RF.columns[index]] < eval_pval["BH_value"]]
    
    return eval_pval , eval_pd



def extract_ANN_data_for_plot(ANN_total_data,DS):
    '''
    Function to extract predictions and observations from ANN results.
    Output: Pd dataframe with columns: [nncno, Type (pred/obs), variable (PK), value, Fold]
    
    '''

    scatter_long_pd = pd.DataFrame()
    for i in range(5):

        y_pred = ANN_total_data[DS][i]["output_tensor"].assign(Type = "predictions")
        y_target = ANN_total_data[DS][i]["True_Y"].assign(Type = "observations")
        scatter_long_tmp = pd.concat([y_pred,y_target],axis=0).reset_index().melt(id_vars = ["nncno","Type"]).assign(Fold = i)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
    return scatter_long_pd


def extract_ANN_data_for_plot_original_space(ANN_total_data,DS):
    '''
    Function to extract predictions and observations from ANN results.
    Output: Pd dataframe with columns: [nncno, Type (pred/obs), variable (PK), value, Fold]
    
    '''
    
    
    
    scatter_long_pd = pd.DataFrame()
    for i in range(5):

        y_pred = ANN_total_data[DS][i]["output_tensor"]

        y_pred = pd.DataFrame(y_pred)
        y_pred = np.exp(y_pred)
        y_pred = y_pred.assign(Type = "predictions")
        y_pred["index"] = range(0,y_pred.shape[0])

        y_target = ANN_total_data[DS][i]["True_Y"]
        y_target = pd.DataFrame(y_target)
        y_target = np.exp(y_target)
        y_target = y_target.assign(Type = "observations")
        y_target["index"] = range(0,y_target.shape[0])

        scatter_long_tmp = pd.concat([y_pred,y_target],axis=0)
        scatter_long_tmp.columns = ["CL","T1/2","MRT","Type","index"]
        scatter_long_tmp = scatter_long_tmp.melt(id_vars = ["index","Type"])
        scatter_long_tmp["Fold"] = i

        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
        
    return scatter_long_pd


def extract_ANN_data_for_plot_ts(ANN_total_data,DS):
    '''
    Function to extract predictions and observations from ANN results. (temporal - split)
    Output: Pd dataframe with columns: [nncno, Type (pred/obs), variable (PK), value, Fold]
    
    '''

    scatter_long_pd = pd.DataFrame()
    for i in range(5):

        y_pred = ANN_total_data[DS][i]["output_tensor"].assign(Type = "predictions")
        y_target = ANN_total_data[DS][i]["True_Y"].assign(Type = "observations")
        scatter_long_tmp = pd.concat([y_pred,y_target],axis=0).reset_index().melt(id_vars = ["nncno","nncno_project","Type"]).assign(Fold = i)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
    return scatter_long_pd

def extract_ANN_data_for_plot_ts_DS1(ANN_total_data,DS):
    '''
    Function to extract predictions and observations from ANN results. (temporal - split)
    Output: Pd dataframe with columns: [nncno, Type (pred/obs), variable (PK), value, Fold]
    
    '''

    scatter_long_pd = pd.DataFrame()
    for i in range(5):

        y_pred = ANN_total_data[DS][i]["output_tensor"].assign(Type = "predictions")
        y_target = ANN_total_data[DS][i]["True_Y"].assign(Type = "observations")
        scatter_long_tmp = pd.concat([y_pred,y_target],axis=0).reset_index().melt(id_vars = ["UploadDate","Type"]).assign(Fold = i)
        scatter_long_pd = pd.concat([scatter_long_pd,scatter_long_tmp],axis=0)
    return scatter_long_pd


# In[8]:


ANN_total_data, ANN_rmse_data, ANN_rmse_data_CL, ANN_rmse_data_T12, ANN_rmse_data_MRT = store_ANN_data()


# # Figure 3 (CL, T12, MRT)

# RF data
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_CL = RMSE_random_forest[["T1/2[h]","DS"]]
RMSE_random_forest_CL["Model"] = "Random Forest"
RMSE_random_forest_CL.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd = pd.DataFrame(ANN_rmse_data_T12)
ANN_outer_test_scores_long = ANN_outer_test_scores_pd.melt()
ANN_outer_test_scores_long["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_CL,ANN_outer_test_scores_long],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))


ax1, ax2 = g.axes[0]

ax1.axhline(1.44, ls='--',color = "black")
ax1.text(10.7,1.39, "Mean predictor model",size=17)
ax2.axhline(1.44, ls='--',color = "black")
ax2.text(10.7,1.39, "Mean predictor model",size=17)


ax1.axhline(0.16, ls='--',color = "black")
ax1.text(10.7,0.19, "Experimental error",size=17)
ax2.axhline(0.16, ls='--',color = "black")
ax2.text(10.7,0.19, "Experimental error",size=17)


[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)

g.savefig('../reports/figures/Figure3_top_T12.png', bbox_inches='tight') 


## Zoomed in area (black box)
ANN_outer_test_scores_reduced = ANN_outer_test_scores_pd.loc[:, ANN_outer_test_scores_pd.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN_T12.png', bbox_inches='tight') 


RMSE_random_forest_CL_reduced = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134", "DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_CL_reduced, kind="box",order=sorted(RMSE_random_forest_CL_reduced.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_CL_reduced.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF_T12.png', bbox_inches='tight') 



# RF data
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_CL = RMSE_random_forest[["MRT[h]","DS"]]
RMSE_random_forest_CL["Model"] = "Random Forest"
RMSE_random_forest_CL.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd = pd.DataFrame(ANN_rmse_data_MRT)
ANN_outer_test_scores_long = ANN_outer_test_scores_pd.melt()
ANN_outer_test_scores_long["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_CL,ANN_outer_test_scores_long],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)

ax1, ax2 = g.axes[0]

ax1.axhline(1.41, ls='--',color = "black")
ax1.text(10.7,1.35, "Mean predictor model",size=17)
ax2.axhline(1.41, ls='--',color = "black")
ax2.text(10.7,1.35, "Mean predictor model",size=17)



ax1.axhline(0.11,ls='--',color = "black")
ax1.text(10.7,0.15, "Experimental error",size=17)
ax2.axhline(0.11,ls='--',color = "black")
ax2.text(10.7,0.15, "Experimental error",size=17)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)

g.savefig('../reports/figures/Figure3_top_MRT.png', bbox_inches='tight') 


# In[80]:


RMSE_random_forest_CL_reduced = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134", "DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_CL_reduced, kind="box",order=sorted(RMSE_random_forest_CL_reduced.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_CL_reduced.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF_MRT.png', bbox_inches='tight') 


# In[79]:


## Zoomed in area (black box)
ANN_outer_test_scores_reduced = ANN_outer_test_scores_pd.loc[:, ANN_outer_test_scores_pd.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN_MRT.png', bbox_inches='tight') 


# In[49]:


# RF data
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_CL = RMSE_random_forest[["CL[ml/min/kg]","DS"]]


RMSE_random_forest_CL["Model"] = "Random Forest"
RMSE_random_forest_CL.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd = pd.DataFrame(ANN_rmse_data_CL)
ANN_outer_test_scores_long = ANN_outer_test_scores_pd.melt()
ANN_outer_test_scores_long["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_CL,ANN_outer_test_scores_long],axis=0)

## PLotting
sns.set(font_scale = 2)
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))


ax1, ax2 = g.axes[0]

ax1.axhline(1.38, ls='--',color="black")
ax1.text(10.7,1.33, "Mean predictor model",size=17)
ax2.axhline(1.38, ls='--',color="black")
ax2.text(10.7,1.33, "Mean predictor model",size=17)



ax1.axhline(0.148, ls='--',color="black")
ax1.text(11,0.16, "Experimental error",size=17)
ax2.axhline(0.148, ls='--',color="black")
ax2.text(11,0.16, "Experimental error",size=17)


[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]

g.savefig('../reports/figures/Figure3_top_CL.png', bbox_inches='tight') 


DS2_ANN_numeric = pd.DataFrame(ANN_rmse_data_CL["DS2"],columns = ["value"])
DS2_ANN_numeric["variable"] = "DS2 descriptors"
DS2_ANN_numeric["Model"] = "ANN"

DS2_ANN_ecfp4 = pd.DataFrame(pickle.load(open('../data/processed/ANN_ds2_results.pkl','rb')),columns = ["value"])
DS2_ANN_ecfp4["variable"] = "DS2 ecfp4"
DS2_ANN_ecfp4["Model"] = "ANN"


DS2_RF_numeric = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"] == "DS2"].reset_index().drop("Fold",axis=1)
DS2_RF_ecfp4 = pd.DataFrame(pickle.load(open('../data/processed/Random_forest_ecfp4_ds2_results.pkl','rb')),columns = ["value"])


DS2_RF_numeric["variable"] = "DS2 descriptors"
DS2_RF_ecfp4["variable"] = "DS2 ecfp4"
DS2_RF_ecfp4["Model"] = "Random Forest"

total_for_plot = pd.concat([DS2_RF_numeric,DS2_RF_ecfp4,DS2_ANN_numeric,DS2_ANN_ecfp4],axis=0)


## PLotting
g = sns.catplot(x="variable", y="value", data=total_for_plot, kind="box",col = "Model",palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k')
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',yticks = np.arange(0.4,1.35,0.1))
g.set(ylim=(0.4, 1.35))

ax1, ax2 = g.axes[0]

g.savefig('../reports/figures/Supporting_figure_ECFP4_DS2.png', bbox_inches='tight') 


# # Figure 3 (zoomed)

# In[50]:


## Zoomed in area (black box)
ANN_outer_test_scores_reduced = ANN_outer_test_scores_pd.loc[:, ANN_outer_test_scores_pd.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.3,1,0.05))
g.set(ylim=(0.3, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN.png', bbox_inches='tight') 


# In[51]:


RMSE_random_forest_CL_reduced = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134", "DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_CL_reduced, kind="box",order=sorted(RMSE_random_forest_CL_reduced.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_CL_reduced.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.3,1,0.05))
g.set(ylim=(0.3, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF.png', bbox_inches='tight') 


# # Figure 4

# In[ ]:


## Calculate paired t-test and visualize in heatmap on Random Forest
RMSE_random_forest_CL_reduced_wide = RMSE_random_forest_CL_reduced.reset_index().pivot(index = "Fold",columns = "variable",values = "value")
RMSE_random_forest_CL_reduced_wide = RMSE_random_forest_CL_reduced_wide[ANN_outer_test_scores_reduced.columns]

t_test_scores_RF = paired_t_test_values(RMSE_random_forest_CL_reduced_wide)
#plot_t_test_values(t_test_scores_RF,"Paired t-test for Random Forest models")

## Calculate paired t-test and visualize in heatmap on ANN
t_test_scores_ANN = paired_t_test_values(ANN_outer_test_scores_reduced)
#plot_t_test_values(t_test_scores_ANN,"Paired t-test for ANN models")

# Calculate the paired t-test difference between model on same descriptorsets
t_test_diff = paired_t_test_across_model(RMSE_random_forest_CL_reduced_wide,ANN_outer_test_scores_reduced)
pd.DataFrame([RMSE_random_forest_CL_reduced_wide.columns,t_test_diff]).T
#pd.DataFrame(t_test_diff)


# In[20]:


t_test_scores_RF


# In[21]:


calculate_BH(t_test_scores_RF,7)


# In[22]:


sns.set_theme(style="white")

t_test_RF = t_test_scores_RF
t_test_ANN = t_test_scores_ANN
mask_RF= np.tril(np.ones_like(t_test_RF, dtype=bool))
mask_ANN = np.triu(np.ones_like(t_test_ANN, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(t_test_RF, mask=mask_RF, cmap='Blues', vmax=1, vmin=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .85, "pad":-.01}, ax=ax,annot=True)

#flare
sns.heatmap(t_test_ANN, mask=mask_ANN, cmap='flare', vmax=1, vmin=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .85}, ax=ax,annot=True)
# Manually:
# the following lines color and hatch the axes background, only the diagonals are visible
ax.patch.set_facecolor('darkgrey')
#ax.patch.set_edgecolor('yellow')
#ax.patch.set_hatch('xx')
ax.set_title('P values from paired t-tests',fontsize=15)
ax.text(8.3, -0.15,'ANN', fontsize=17)
ax.text(9.9, -0.15,'RF', fontsize=17)

## Add black boxes in figure 4 manually:
ax.add_patch(Rectangle((4,0),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((4,2),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((7,2),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((7,3),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((7,5),1,1, fill=False, edgecolor='black', lw=4))


ax.add_patch(Rectangle((0,3),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((0,4),1,1, fill=False, edgecolor='black', lw=4))

ax.add_patch(Rectangle((0,6),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((1,3),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((1,4),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((3,7),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((5,7),1,1, fill=False, edgecolor='black', lw=4))
ax.add_patch(Rectangle((6,7),1,1, fill=False, edgecolor='black', lw=4))

# Check boxes before exporting
plt.savefig("../reports/figures/Figure4.png",bbox_inches = 'tight') 


# # Figure 5(s)

# In[23]:


plt.style.use('ggplot')
Random_forest_scatter = pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
scatter_1 = Random_forest_scatter["DS1"].assign(Model = "DS1(RF)")
scatter_124 = Random_forest_scatter["DS124"].assign(Model = "DS124(RF)")
scatter_random_forest_concat = pd.concat([scatter_1,scatter_124],axis=0)
## Make ANN data:
scatter_1_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1").assign(Model = "DS1(ANN)")
scatter_1234_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1234").assign(Model = "DS1234(ANN)")
scatter_ANN_concat = pd.concat([scatter_1_ANN,scatter_1234_ANN],axis=0)

# Concat to long with both RF and ANN
total_long = pd.concat([scatter_random_forest_concat,scatter_ANN_concat],axis=0)


data_for_t12_plot = total_long[total_long.variable == "CL[ml/min/kg]"].drop(["variable","Fold"],axis=1)
data_for_t12_plot = data_for_t12_plot.pivot(index = ["Model","nncno"],columns = "Type",values = "value").reset_index().set_index("nncno")
data_for_t12_plot["Model_type"] = np.where(data_for_t12_plot.Model.str[-4:] == "(RF)","RF","ANN")


data_for_t12_plot_original_domain = data_for_t12_plot.copy()
data_for_t12_plot_original_domain["observations"] = np.exp(data_for_t12_plot_original_domain["observations"])
data_for_t12_plot_original_domain["predictions"] = np.exp(data_for_t12_plot_original_domain["predictions"])


fig, axs = plt.subplots(2, 2,figsize=(10,8),dpi = 400,sharex=True, sharey=True)
idx_name = ["DS1(RF)","DS124(RF)","DS1(ANN)","DS1234(ANN)"]
kk = 0
RMSE_R2_t12 = np.round(data_for_t12_plot.groupby(["Model"]).apply( r2_rmse_updated ),2).reset_index()
RMSE_R2_t12.set_index("Model",inplace=True)
RMSE_R2_t12 = RMSE_R2_t12.reindex(idx_name)

aspart_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0014"]
aspart_data_original_domain = data_for_t12_plot_original_domain[data_for_t12_plot_original_domain.index == "0121-0000-0014"]

degludec_data = data_for_t12_plot[data_for_t12_plot.index == "0100-0000-0454"]
degludec_data_original_domain = data_for_t12_plot_original_domain[data_for_t12_plot_original_domain.index == "0100-0000-0454"]

human_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0308"]
human_data_original_domain = data_for_t12_plot_original_domain[data_for_t12_plot_original_domain.index == "0121-0000-0308"]

LP_data = data_for_t12_plot[data_for_t12_plot.index == "0123-0000-0327"]
LP_data_original_domain = data_for_t12_plot_original_domain[data_for_t12_plot_original_domain.index == "0123-0000-0327"]


for i in range(2):
    for j in range(2):
           #axs[i,j].set(xscale="log",yscale="log")
            axs[i,j].plot(data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["observations"], data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["predictions"],'o',ms=1.5,color = "gray")
            axs[i,j].plot(aspart_data[aspart_data.Model == idx_name[kk]]["observations"], aspart_data[aspart_data.Model == idx_name[kk]]["predictions"],'o',ms=7,color = "darkred")
            axs[i,j].plot(degludec_data[degludec_data.Model == idx_name[kk]]["observations"], degludec_data[degludec_data.Model == idx_name[kk]]["predictions"],'v',ms=7,color = "darkgreen")
            axs[i,j].plot(human_data[human_data.Model == idx_name[kk]]["observations"], human_data[human_data.Model == idx_name[kk]]["predictions"],'P',ms=7,color = "darkblue")
            axs[i,j].plot(LP_data[LP_data.Model == idx_name[kk]]["observations"], LP_data[LP_data.Model == idx_name[kk]]["predictions"],'^',ms=7,color = "darkorange")
            
            axs[i,j].axline((1, 1), slope=1,color="black", dashes=(3, 2),linewidth=0.8)
            
            xx = np.linspace(0.01, 60, 1000)
            xx_fold2_upper = xx*2
            xx_fold2_lower = xx*0.5
            
            
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_upper),color = "darkred",linestyle="--",linewidth=0.8)
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_lower),color = "darkred",linestyle="--",linewidth=0.8)
            
            axs[i,j].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE =RMSE_R2_t12.iloc[kk,0], R2 = RMSE_R2_t12.iloc[kk,1]),fontsize=12)
                
            
            axs[i,j].tick_params(axis='both', which='major', labelsize=12) 
            #xlabel("X axis label")
            #fig[i,j].ylabel("Y axis label")
            
            kk = kk +1 
           
axs[0,0].text(-1.6, 6.5,'DS1(RF)', fontsize=17)
axs[0,1].text(-2, 6.5,'DS124(RF)', fontsize=17)
axs[1,0].text(-1.9, 6.5,'DS1(ANN)', fontsize=17)
axs[1,1].text(-2.4, 6.5,'DS1234(ANN)', fontsize=17)
# add zeros to maintain same number of significant digits.
axs[0,1].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE = "0.60",R2 = "0.82"),fontsize=12)
axs[1,0].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE = "0.90",R2 = "0.59"),fontsize=12)

axs[0,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,1].set_xlabel('Observations (log)', fontsize=15)
axs[1,0].set_xlabel('Observations (log)', fontsize=15)

legend_elements = [Line2D([0], [0], marker='o', color = "lightgray", label='Insulin Aspart',markersize = 10,markerfacecolor = "darkred"),
                   Line2D([0], [0], marker='v', color='lightgray', label='Insulin Degludec', markersize=10,markerfacecolor = "darkgreen"),
                   Line2D([0], [0], marker='P', color='lightgray', label='Human Insulin', markersize=10,markerfacecolor = "darkblue"),
                   Line2D([0], [0], marker='^', color='lightgray', label='Insulin-0327', markersize=10,markerfacecolor = "darkorange", )
                   
                  ]
axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(-0.15, 2.55), ncol=2,prop={'size': 15})
#axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(1.55, 1))

plt.savefig("../reports/figures/Figure5_CL_scatter.png",bbox_inches = 'tight') 


#pos_list = [0.001,0.1,1,10,100]                  
#name_list_2 = [-4,-2,0,2,4]
#axs[i,j].xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
#axs[i,j].yaxis.set_major_locator(ticker.FixedLocator((pos_list)))

#axs[i,j].xaxis.set_major_formatter(ticker.FixedFormatter((name_list_2)))
#axs[i,j].yaxis.set_major_formatter(ticker.FixedFormatter((name_list_2)))



# In[24]:


plt.figure(facecolor='lightgray')

plt.plot(data_for_t12_plot[data_for_t12_plot.Model == "DS124(RF)"]["observations"], data_for_t12_plot[data_for_t12_plot.Model == "DS124(RF)"]["predictions"],'o',ms=1.5,color = "gray")
plt.plot(aspart_data[aspart_data.Model == "DS124(RF)"]["observations"], aspart_data[aspart_data.Model == "DS124(RF)"]["predictions"],'o',ms=7,color = "darkred")
plt.plot(degludec_data[degludec_data.Model == "DS124(RF)"]["observations"], degludec_data[degludec_data.Model == "DS124(RF)"]["predictions"],'v',ms=7,color = "darkgreen")
plt.plot(human_data[human_data.Model == "DS124(RF)"]["observations"], human_data[human_data.Model == "DS124(RF)"]["predictions"],'P',ms=7,color = "darkblue")
plt.plot(LP_data[LP_data.Model == "DS124(RF)"]["observations"], LP_data[LP_data.Model == "DS124(RF)"]["predictions"],'^',ms=7,color = "darkorange")

plt.axline((1, 1), slope=1,color="black", dashes=(3, 2),linewidth=0.8)

# Plot 2 fold error in figure 
xx = np.linspace(0.01, 60, 1000)
xx_fold2_upper = xx*2
xx_fold2_lower = xx*0.5
  

plt.plot(np.log(xx), np.log(xx_fold2_upper),color = "darkred",linestyle="--",linewidth=0.8)
plt.plot(np.log(xx), np.log(xx_fold2_lower),color = "darkred",linestyle="--",linewidth=0.8)

ax.set_ylabel('Predictions (log)', fontsize=15)
ax.set_xlabel('Observations (log)', fontsize=15)

plt.savefig("../reports/figures/TOC_graphic.png",bbox_inches = 'tight',dpi = 500) 


# In[25]:


# T1/2
plt.style.use('ggplot')
Random_forest_scatter = pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
scatter_1 = Random_forest_scatter["DS1"].assign(Model = "DS1(RF)")
scatter_124 = Random_forest_scatter["DS124"].assign(Model = "DS124(RF)")
scatter_random_forest_concat = pd.concat([scatter_1,scatter_124],axis=0)
## Make ANN data:
scatter_1_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1").assign(Model = "DS1(ANN)")
scatter_1234_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1234").assign(Model = "DS1234(ANN)")
scatter_ANN_concat = pd.concat([scatter_1_ANN,scatter_1234_ANN],axis=0)

# Concat to long with both RF and ANN
total_long = pd.concat([scatter_random_forest_concat,scatter_ANN_concat],axis=0)


data_for_t12_plot = total_long[total_long.variable == "T1/2[h]"].drop(["variable","Fold"],axis=1)
data_for_t12_plot = data_for_t12_plot.pivot(index = ["Model","nncno"],columns = "Type",values = "value").reset_index().set_index("nncno")
data_for_t12_plot["Model_type"] = np.where(data_for_t12_plot.Model.str[-4:] == "(RF)","RF","ANN")


fig, axs = plt.subplots(2, 2,figsize=(10,8),dpi = 400,sharex=True, sharey=True)
idx_name = ["DS1(RF)","DS124(RF)","DS1(ANN)","DS1234(ANN)"]
kk = 0

RMSE_R2_t12 = np.round(data_for_t12_plot.groupby(["Model"]).apply( r2_rmse_updated ),2).reset_index()
RMSE_R2_t12.set_index("Model",inplace=True)
RMSE_R2_t12 = RMSE_R2_t12.reindex(idx_name)


aspart_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0014"]
degludec_data = data_for_t12_plot[data_for_t12_plot.index == "0100-0000-0454"]
human_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0308"]
LP_data = data_for_t12_plot[data_for_t12_plot.index == "0123-0000-0327"]


for i in range(2):
    for j in range(2):
            axs[i,j].plot(data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["observations"], data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["predictions"],'o',ms=1.5,color = "gray")
            axs[i,j].plot(aspart_data[aspart_data.Model == idx_name[kk]]["observations"], aspart_data[aspart_data.Model == idx_name[kk]]["predictions"],'o',ms=7,color = "darkred")
            axs[i,j].plot(degludec_data[degludec_data.Model == idx_name[kk]]["observations"], degludec_data[degludec_data.Model == idx_name[kk]]["predictions"],'v',ms=7,color = "darkgreen")
            axs[i,j].plot(human_data[human_data.Model == idx_name[kk]]["observations"], human_data[human_data.Model == idx_name[kk]]["predictions"],'P',ms=7,color = "darkblue")
            axs[i,j].plot(LP_data[LP_data.Model == idx_name[kk]]["observations"], LP_data[LP_data.Model == idx_name[kk]]["predictions"],'^',ms=7,color = "darkorange")
            axs[i,j].axline((1, 1), slope=1,color="black", dashes=(3, 2),linewidth=0.8)
            
            xx = np.linspace(0.08, 150, 1000)
            xx_fold2_upper = xx*2
            xx_fold2_lower = xx*0.5
            
            
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_upper),color = "darkred",linestyle="--",linewidth=0.8)
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_lower),color = "darkred",linestyle="--",linewidth=0.8)
            
            axs[i,j].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE =RMSE_R2_t12.iloc[kk,0], R2 = RMSE_R2_t12.iloc[kk,1]),fontsize=12)
            axs[i,j].tick_params(axis='both', which='major', labelsize=12) 
            #xlabel("X axis label")
            #fig[i,j].ylabel("Y axis label")
            
            kk = kk +1 
           
axs[0,0].text(-0.1, 7.3,'DS1(RF)', fontsize=17)
axs[0,1].text(0, 7.3,'DS124(RF)', fontsize=17)
axs[1,0].text(-0.2, 7.3,'DS1(ANN)', fontsize=17)
axs[1,1].text(-0.8, 7.3,'DS1234(ANN)', fontsize=17)

axs[0,1].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE = "0.50", R2 = "0.80"),fontsize=12)
axs[1,0].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE = "0.70", R2 = "0.61"),fontsize=12)

axs[0,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,1].set_xlabel('Observations (log)', fontsize=15)
axs[1,0].set_xlabel('Observations (log)', fontsize=15)

legend_elements = [Line2D([0], [0], marker='o', color = "lightgray", label='Insulin Aspart',markersize = 10,markerfacecolor = "darkred"),
                   Line2D([0], [0], marker='v', color='lightgray', label='Insulin Degludec', markersize=10,markerfacecolor = "darkgreen"),
                   Line2D([0], [0], marker='P', color='lightgray', label='Human Insulin', markersize=10,markerfacecolor = "darkblue"),
                   Line2D([0], [0], marker='^', color='lightgray', label='Insulin-0327', markersize=10,markerfacecolor = "darkorange", )
                   
                  ]
axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(-0.15, 2.55), ncol=2,prop={'size': 15})
#axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(1.55, 1))

plt.savefig("../reports/figures/Figure5_T12[h]_scatter.png",bbox_inches = 'tight') 


# In[26]:


# MRT
plt.style.use('ggplot')
Random_forest_scatter = pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
scatter_1 = Random_forest_scatter["DS1"].assign(Model = "DS1(RF)")
scatter_124 = Random_forest_scatter["DS124"].assign(Model = "DS124(RF)")
scatter_random_forest_concat = pd.concat([scatter_1,scatter_124],axis=0)
## Make ANN data:
scatter_1_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1").assign(Model = "DS1(ANN)")
scatter_1234_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1234").assign(Model = "DS1234(ANN)")
scatter_ANN_concat = pd.concat([scatter_1_ANN,scatter_1234_ANN],axis=0)

# Concat to long with both RF and ANN
total_long = pd.concat([scatter_random_forest_concat,scatter_ANN_concat],axis=0)


data_for_t12_plot = total_long[total_long.variable == "MRT[h]"].drop(["variable","Fold"],axis=1)
data_for_t12_plot = data_for_t12_plot.pivot(index = ["Model","nncno"],columns = "Type",values = "value").reset_index().set_index("nncno")
data_for_t12_plot["Model_type"] = np.where(data_for_t12_plot.Model.str[-4:] == "(RF)","RF","ANN")


fig, axs = plt.subplots(2, 2,figsize=(10,8),dpi = 400,sharex=True, sharey=True)
idx_name = ["DS1(RF)","DS124(RF)","DS1(ANN)","DS1234(ANN)"]
kk = 0
RMSE_R2_t12 = np.round(data_for_t12_plot.groupby(["Model"]).apply( r2_rmse_updated ),2).reset_index()
RMSE_R2_t12.set_index("Model",inplace=True)
RMSE_R2_t12 = RMSE_R2_t12.reindex(idx_name)

#aspart_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0014"]
degludec_data = data_for_t12_plot[data_for_t12_plot.index == "0100-0000-0454"]
human_data = data_for_t12_plot[data_for_t12_plot.index == "0121-0000-0308"]
LP_data = data_for_t12_plot[data_for_t12_plot.index == "0123-0000-0327"]


for i in range(2):
    for j in range(2):
            axs[i,j].plot(data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["observations"], data_for_t12_plot[data_for_t12_plot.Model == idx_name[kk]]["predictions"],'o',ms=1.5,color = "gray")
            axs[i,j].plot(degludec_data[degludec_data.Model == idx_name[kk]]["observations"], degludec_data[degludec_data.Model == idx_name[kk]]["predictions"],'v',ms=7,color = "darkgreen")
            axs[i,j].plot(human_data[human_data.Model == idx_name[kk]]["observations"], human_data[human_data.Model == idx_name[kk]]["predictions"],'P',ms=7,color = "darkblue")
            axs[i,j].plot(LP_data[LP_data.Model == idx_name[kk]]["observations"], LP_data[LP_data.Model == idx_name[kk]]["predictions"],'^',ms=7,color = "darkorange")
            
            axs[i,j].axline((1, 1), slope=1,color="black", dashes=(3, 2),linewidth=0.8)
            
            
            xx = np.linspace(0.08, 150, 1000)
            xx_fold2_upper = xx*2
            xx_fold2_lower = xx*0.5
            
            
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_upper),color = "darkred",linestyle="--",linewidth=0.8)
            axs[i,j].plot(np.log(xx), np.log(xx_fold2_lower),color = "darkred",linestyle="--",linewidth=0.8)
            
            axs[i,j].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE =RMSE_R2_t12.iloc[kk,0], R2 = RMSE_R2_t12.iloc[kk,1]),fontsize=12)
            axs[i,j].tick_params(axis='both', which='major', labelsize=12) 
            #xlabel("X axis label")
            #fig[i,j].ylabel("Y axis label")
            
            kk = kk +1 
           
axs[0,0].text(-0.1, 7.3,'DS1(RF)', fontsize=17)
axs[0,1].text(-0, 7.3,'DS124(RF)', fontsize=17)
axs[1,0].text(-0.2, 7.3,'DS1(ANN)', fontsize=17)
axs[1,1].text(-0.8, 7.3,'DS1234(ANN)', fontsize=17)

axs[1,1].set_title("RMSE = {RMSE}, R2 = {R2}".format(RMSE = "0.55", R2 = "0.80"),fontsize=12)

axs[0,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,0].set_ylabel('Predictions (log)', fontsize=15)
axs[1,1].set_xlabel('Observations (log)', fontsize=15)
axs[1,0].set_xlabel('Observations (log)', fontsize=15)

legend_elements = [
                   Line2D([0], [0], marker='v', color='lightgray', label='Degludec', markersize=10,markerfacecolor = "darkgreen"),
                   Line2D([0], [0], marker='P', color='lightgray', label='Human Insulin', markersize=10,markerfacecolor = "darkblue"),
                   Line2D([0], [0], marker='^', color='lightgray', label='Insulin-0327', markersize=10,markerfacecolor = "darkorange", )
                   
                  ]
axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(-0.15, 2.55), ncol=2,prop={'size': 15})
#axs[1][1].legend(handles=legend_elements,loc = "center",bbox_to_anchor=(1.55, 1))

plt.savefig("../reports/figures/Figure5_MRT_scatter.png",bbox_inches = 'tight') 


# # Figure S5

# In[27]:


# RF data
sns.set(font_scale=2)
RMSE_random_forest_CL_reduced = RMSE_random_forest_CL_reduced[RMSE_random_forest_CL_reduced["variable"].isin(["DS123","DS1234","DS124","DS134","DS14","DS13"])]
RMSE_random_forest_CL_reduced["PCA"]= "Yes"

Random_scatter_NO_PCA= pickle.load(open('../data/processed/Random_forest_scatter_file_No_PCA.pkl','rb'))
RMSE_random_forest_NO_PCA, R2_random_forest_NO_PCA = extract_RMSE_R2_RF(Random_scatter_NO_PCA)
RMSE_random_forest_CL_NO_PCA = RMSE_random_forest_NO_PCA[["CL[ml/min/kg]","DS"]]
RMSE_random_forest_CL_NO_PCA["Model"] = "Random Forest"
RMSE_random_forest_CL_NO_PCA.columns = ["value","variable","Model"]
RMSE_random_forest_CL_NO_PCA = RMSE_random_forest_CL_NO_PCA[RMSE_random_forest_CL_NO_PCA["variable"].isin(["DS123","DS1234","DS124","DS134","DS14","DS13"])]
RMSE_random_forest_CL_NO_PCA["PCA"] = "No"

full_for_plot_RF_PCA = pd.concat([RMSE_random_forest_CL_reduced,RMSE_random_forest_CL_NO_PCA],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=full_for_plot_RF_PCA, kind="box",col = "PCA",palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k')
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',yticks = np.arange(0.45,1,0.1))
g.set(ylim=(0.45, 1))

g.savefig('../reports/figures/Figure_S5.png', bbox_inches='tight') 


def extract_RMSE_R2_RF(Random_forest_scatter):
    Descriptor_combinations = ["DS1","DS12","DS123","DS1234","DS124","DS134","DS14","DS13","DS2","DS3","DS4","DS23","DS24","DS234","DS34"]
    
    RMSE_total = pd.DataFrame()
    R2_total = pd.DataFrame()
    
    for des in Descriptor_combinations:
        tmp = Random_forest_scatter[des].pivot(index = ["nncno","variable","Fold"],columns = "Type",values = "value").reset_index().set_index("nncno")
        RMSE_table = tmp.groupby(["Fold","variable"]).apply(rmse_2).reset_index()
        RMSE_table.columns = ['Fold', 'variable', 'value']
        RMSE_table = RMSE_table.pivot(index = "Fold",columns = "variable",values = "value")
        RMSE_table["DS"] = des

        R2_table = tmp.groupby(["Fold","variable"]).apply(r2_2).reset_index()
        R2_table.columns = ['Fold', 'variable', 'value']
        R2_table = R2_table.pivot(index = "Fold",columns = "variable",values = "value")
        R2_table["DS"] = des
        
        RMSE_total = pd.concat([RMSE_total,RMSE_table],axis=0)
        R2_total = pd.concat([R2_total,R2_table],axis=0)
        
        
    return RMSE_total, R2_total


# In[29]:


ANN_total_data_ts, ANN_rmse_data_ts, ANN_rmse_data_CL_ts, ANN_rmse_data_T12_ts, ANN_rmse_data_MRT_ts = store_ANN_data_ts()


# In[52]:


# RF data
sns.set(font_scale=2)
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file_ts.pkl','rb'))


RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_CL_ts = RMSE_random_forest[["CL[ml/min/kg]","DS"]]


RMSE_random_forest_CL_ts["Model"] = "Random Forest"
RMSE_random_forest_CL_ts.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd_ts = pd.DataFrame(ANN_rmse_data_CL_ts)
ANN_outer_test_scores_long_ts = ANN_outer_test_scores_pd_ts.melt()
ANN_outer_test_scores_long_ts["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_CL_ts,ANN_outer_test_scores_long_ts],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))


ax1, ax2 = g.axes[0]


ax1.axhline(1.44, ls='--',color="black")
ax1.text(10.7,1.38, "Mean predictor model",size=17)
ax2.axhline(1.44, ls='--',color="black")
ax2.text(10.7,1.38, "Mean predictor model",size=17)



ax1.axhline(0.148, ls='--',color="black")
ax1.text(11,0.16, "Experimental error",size=17)
ax2.axhline(0.148, ls='--',color="black")
ax2.text(11,0.16, "Experimental error",size=17)

[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)
g.savefig('../reports/figures/Figure_S11_top_CL.png', bbox_inches='tight') 


# In[53]:


sns.set(font_scale=2)
RMSE_random_forest_CL_reduced_ts = RMSE_random_forest_CL_ts[RMSE_random_forest_CL_ts["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134","DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_CL_reduced_ts, kind="box",order=sorted(RMSE_random_forest_CL_reduced_ts.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_CL_reduced_ts.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.3,1,0.05))
g.set(ylim=(0.3, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF_ts_split_CL.png', bbox_inches='tight') 


# In[54]:


## Zoomed in area (black box)
sns.set(font_scale=2)
ANN_outer_test_scores_reduced_ts = ANN_outer_test_scores_pd_ts.loc[:, ANN_outer_test_scores_pd_ts.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced_ts.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.3,1,0.05))
g.set(ylim=(0.3, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN_ts_split_CL.png', bbox_inches='tight') 


# In[97]:


# RF data
sns.set(font_scale=2)
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file_ts.pkl','rb'))


RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_T12_ts = RMSE_random_forest[["MRT[h]","DS"]]


RMSE_random_forest_T12_ts["Model"] = "Random Forest"
RMSE_random_forest_T12_ts.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd_ts = pd.DataFrame(ANN_rmse_data_T12_ts)
ANN_outer_test_scores_long_ts = ANN_outer_test_scores_pd_ts.melt()
ANN_outer_test_scores_long_ts["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_T12_ts,ANN_outer_test_scores_long_ts],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))


ax1, ax2 = g.axes[0]


ax1.axhline(1.42, ls='--',color="black")
ax1.text(10.7,1.37, "Mean predictor model",size=17)
ax2.axhline(1.42, ls='--',color="black")
ax2.text(10.7,1.37, "Mean predictor model",size=17)



ax1.axhline(0.128, ls='--',color="black")
ax1.text(11,0.16, "Experimental error",size=17)
ax2.axhline(0.128, ls='--',color="black")
ax2.text(11,0.16, "Experimental error",size=17)

[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)
g.savefig('../reports/figures/Figure_S11_top_MRT.png', bbox_inches='tight') 


# In[95]:


## Zoomed in area (black box)
sns.set(font_scale=2)
ANN_outer_test_scores_reduced_ts = ANN_outer_test_scores_pd_ts.loc[:, ANN_outer_test_scores_pd_ts.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced_ts.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN_ts_split_MRT.png', bbox_inches='tight') 


# In[98]:


sns.set(font_scale=2)
RMSE_random_forest_CL_reduced_ts = RMSE_random_forest_T12_ts[RMSE_random_forest_T12_ts["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134","DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_CL_reduced_ts, kind="box",order=sorted(RMSE_random_forest_CL_reduced_ts.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_CL_reduced_ts.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (MRT)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF_ts_split_MRT.png', bbox_inches='tight') 


# In[69]:


# RF data
sns.set(font_scale=2)
Random_scatter= pickle.load(open('../data/processed/Random_forest_scatter_file_ts.pkl','rb'))


RMSE_random_forest, R2_random_forest = extract_RMSE_R2_RF(Random_scatter)
RMSE_random_forest_T12_ts = RMSE_random_forest[["T1/2[h]","DS"]]


RMSE_random_forest_T12_ts["Model"] = "Random Forest"
RMSE_random_forest_T12_ts.columns = ["value","variable","Model"]

# ANN data
ANN_outer_test_scores_pd_ts = pd.DataFrame(ANN_rmse_data_T12_ts)
ANN_outer_test_scores_long_ts = ANN_outer_test_scores_pd_ts.melt()
ANN_outer_test_scores_long_ts["Model"] = "ANN"

# Total data concatenated for plotting
outer_test_data = pd.concat([RMSE_random_forest_T12_ts,ANN_outer_test_scores_long_ts],axis=0)

## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "Model",order=sorted(outer_test_data.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4,)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(outer_test_data.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',yticks = np.arange(0,1.5,0.1))
g.set(ylim=(0, 1.5))


ax1, ax2 = g.axes[0]


ax1.axhline(1.48, ls='--',color="black")
ax1.text(10.7,1.40, "Mean predictor model",size=17)
ax2.axhline(1.48, ls='--',color="black")
ax2.text(10.7,1.40, "Mean predictor model",size=17)



ax1.axhline(0.148, ls='--',color="black")
ax1.text(11,0.18, "Experimental error",size=17)
ax2.axhline(0.148, ls='--',color="black")
ax2.text(11,0.18, "Experimental error",size=17)

[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)
g.savefig('../reports/figures/Figure_S11_top_T12.png', bbox_inches='tight') 


# In[70]:


sns.set(font_scale=2)
RMSE_random_forest_T12_reduced_ts = RMSE_random_forest_T12_ts[RMSE_random_forest_T12_ts["variable"].isin(["DS1","DS12","DS123","DS1234","DS124","DS13","DS134","DS14"])]
g = sns.catplot(x="variable", y="value", data=RMSE_random_forest_T12_reduced_ts, kind="box",order=sorted(RMSE_random_forest_CL_reduced_ts.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(RMSE_random_forest_T12_reduced_ts.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_RF_ts_split_T12.png', bbox_inches='tight') 


# In[71]:


## Zoomed in area (black box)
sns.set(font_scale=2)
ANN_outer_test_scores_reduced_ts = ANN_outer_test_scores_pd_ts.loc[:, ANN_outer_test_scores_pd_ts.columns.str.contains('1')]
ANN_outer_test_scores_reduced_long = ANN_outer_test_scores_reduced_ts.melt()
g = sns.catplot(x="variable", y="value", data=ANN_outer_test_scores_reduced_long, kind="box",order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(ANN_outer_test_scores_reduced_long.variable.unique()))
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (T1/2)',title = "",yticks = np.arange(0.4,1,0.05))
g.set(ylim=(0.4, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.savefig('../reports/figures/Figure3_bottom_ANN_ts_split_T12.png', bbox_inches='tight') 


# # Figure S10

# In[38]:


sns.set(font_scale=2)
Vivo_RF_DS124 = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"] == "DS124"]
Vivo_RF_DS124 = Vivo_RF_DS124.reset_index().drop("Fold",axis=1)
Vivo_ANN_DS1234 = ANN_outer_test_scores_long[ANN_outer_test_scores_long["variable"] == "DS1234"]
Vivo_KNN_DS1234 = pd.DataFrame(pickle.load(open('/home/kyei/Project1_clean/models/RF_Vivo_DS1234_KNN.pkl','rb')),columns = ["value"])
Vivo_KNN_DS1234["value"] = Vivo_KNN_DS1234["value"]
Vivo_KNN_DS1234["Model"] = "KNN (DS1234)"

Vivo_KNN_DS124 = pd.DataFrame(pickle.load(open('/home/kyei/Project1_clean/models/RF_Vivo_DS124_KNN.pkl','rb')),columns = ["value"])
Vivo_KNN_DS124["value"] = Vivo_KNN_DS124["value"]
Vivo_KNN_DS124["Model"] = "KNN (DS124)"

Vivo_ANN_DS1234 = Vivo_ANN_DS1234[["value","variable","Model"]]
Vivo_KNN_DS124["variable"] = "DS3"
for_plot = pd.concat([Vivo_ANN_DS1234, Vivo_RF_DS124, Vivo_KNN_DS124,Vivo_KNN_DS1234],axis=0)
for_plot = for_plot.drop("variable",axis=1)
for_plot.columns = ["value","variable"]
for_plot["Model"] = "Random Forest"
for_plot["variable"] = np.where(for_plot["variable"] == "ANN", "ANN (DS1234)",for_plot["variable"])
for_plot["variable"] = np.where(for_plot["variable"] == "Random Forest", "RF (DS124)",for_plot["variable"])
plt.style.use('ggplot')
sns.set(font_scale=2)
g = sns.catplot(x="variable", y="value", col="Model", data=for_plot, kind="box",order=sorted(for_plot.variable.unique()),palette=sns.color_palette("muted"),height=7)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(for_plot.variable.unique()))
g.set(xlabel='Models ', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.45,1,0.1))
g.set(ylim=(0.45, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]

g.savefig('../reports/figures/Figure_S10_KNN_best_models.png', bbox_inches='tight') 


# In[39]:


sns.set(font_scale=2)
Vivo_RF_DS3 = RMSE_random_forest_CL[RMSE_random_forest_CL["variable"] == "DS3"]
Vivo_RF_DS3 = Vivo_RF_DS3.reset_index().drop("Fold",axis=1)
Vivo_ANN_DS3 = ANN_outer_test_scores_long[ANN_outer_test_scores_long["variable"] == "DS3"]
Vivo_KNN_DS3 = pd.DataFrame(pickle.load(open('/home/kyei/Project1_clean/models/RF_Vivo_DS3_KNN.pkl','rb')),columns = ["value"])
Vivo_ANN_DS3 = Vivo_ANN_DS3[["value","variable","Model"]]
Vivo_KNN_DS3["variable"] = "DS3"
Vivo_KNN_DS3["Model"] = "KNN"
for_plot = pd.concat([Vivo_ANN_DS3, Vivo_RF_DS3, Vivo_KNN_DS3],axis=0)
for_plot = for_plot.drop("variable",axis=1)
for_plot.columns = ["value","variable"]
for_plot["Model"] = "RF"
plt.style.use('ggplot')
sns.set(font_scale=2)
g = sns.catplot(x="variable", y="value", col="Model", data=for_plot, kind="box",order=sorted(for_plot.variable.unique()),palette=sns.color_palette("muted"),height=7)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=sorted(for_plot.variable.unique()))
g.set(xlabel='Models (On DS3) ', ylabel='RMSE (CL)',title = "",yticks = np.arange(0.45,1.5,0.1))
g.set(ylim=(0.45, 1.5))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]

g.savefig('../reports/figures/Figure_S10_KNN.png', bbox_inches='tight') 


# # Figure S12

# In[40]:


sns.set(font_scale=2)
Random_forest_outer_test_scores_no_pca_DS5 = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_DS5.pkl','rb'))
Random_forest_outer_test_scores_pca_DS5 = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_DS5_PCA.pkl','rb'))

Random_forest_outer_test_scores_pca_DS5["DS1"] = RMSE_random_forest_CL_reduced_wide["DS1"]
Random_forest_outer_test_scores_pca_DS5["DS12"] = RMSE_random_forest_CL_reduced_wide["DS12"]

Random_forest_outer_test_scores_no_pca_DS5["DS1"] = RMSE_random_forest_CL_reduced_wide["DS1"]
Random_forest_outer_test_scores_no_pca_DS5["DS12"] = RMSE_random_forest_CL_reduced_wide["DS12"]



Random_forest_outer_test_scores_pd_no_pca = pd.DataFrame(Random_forest_outer_test_scores_no_pca_DS5)
Random_forest_outer_test_scores_long_no_pca = Random_forest_outer_test_scores_pd_no_pca.melt()

Random_forest_outer_test_scores_pd_pca = pd.DataFrame(Random_forest_outer_test_scores_pca_DS5)
Random_forest_outer_test_scores_long_pca = Random_forest_outer_test_scores_pd_pca.melt()

Random_forest_outer_test_scores_long_no_pca["PCA"] = "No"
Random_forest_outer_test_scores_long_pca["PCA"] = "Yes"

# Total data concatenated for plotting
outer_test_data = pd.concat([Random_forest_outer_test_scores_long_no_pca,Random_forest_outer_test_scores_long_pca],axis=0)


## PLotting
g = sns.catplot(x="variable", y="value", data=outer_test_data, kind="box",col = "PCA",order=Random_forest_outer_test_scores_long_pca.variable.unique(),palette=sns.color_palette("muted"),height=8,aspect=1.4)
g.map(sns.swarmplot, 'variable', 'value', color='k', order=outer_test_data.variable.unique())
g.set(xlabel='Descriptor Space (DS)', ylabel='RMSE (CL)',yticks = np.arange(0.45,1,0.1))
g.set(ylim=(0.45, 1))
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
left1, bottom1, width1, height1 = (-16, 0.5, 8, 0.18)
rect1=mpatches.Rectangle((left1,bottom1),width1,height1, 
                        fill=False,
                        color="black",
                       linewidth=3)


left2, bottom2, width2, height2 = (-0.4, 0.5, 8, 0.18)
rect2=mpatches.Rectangle((left2,bottom2),width2,height2, 
                        fill=False,
                        color="black",
                       linewidth=3)
#ax = plt.gca()
#ax.add_patch(rect2)
#for ax in g.axes_dict.items():
#    ax.add_patch(rect1)

g.savefig('../reports/figures/Figure_S12.png', bbox_inches='tight',dpi = 400) 


# # Figure S1

# In[41]:


sns.set(font_scale=2)
Vivo_ANN_DS3_OHE = pickle.load(open('/home/kyei/Project1_clean/data/processed/OHE_data.pkl','rb'))
Vivo_ANN_DS3_Zscale = pickle.load(open('/home/kyei/Project1_clean/data/processed/Zscale_data.pkl','rb'))

Vivo_ANN_DS3_OHE = pd.DataFrame(Vivo_ANN_DS3_OHE,columns = ["value"])
Vivo_ANN_DS3_OHE["Model"] = "ANN"
Vivo_ANN_DS3_OHE["variable"] = "DS3"
Vivo_ANN_DS3_OHE["Encoding"] = "OHE"


Vivo_ANN_DS3_Zscale = pd.DataFrame(Vivo_ANN_DS3_Zscale,columns = ["value"])
Vivo_ANN_DS3_Zscale["Model"] = "ANN"
Vivo_ANN_DS3_Zscale["variable"] = "DS3"
Vivo_ANN_DS3_Zscale["Encoding"] = "Z-scale"

Vivo_ANN_DS3_ESM = Vivo_ANN_DS3
Vivo_ANN_DS3_ESM["Encoding"] = "ESM-1b"
for_plot = pd.concat([Vivo_ANN_DS3_OHE,Vivo_ANN_DS3_Zscale,Vivo_ANN_DS3_ESM],axis=0)


# make boxplot with data from the OP
plt.style.use('ggplot')

g = sns.catplot(x="Encoding", y="value", col="Model", data=for_plot, kind="box", height=5, aspect=1.5)
g.map(sns.swarmplot, 'Encoding', 'value', color='k')
g.set(xlabel='Encoding', ylabel='RMSE (CL)')
g.savefig('../reports/figures/Figure_S1.png', bbox_inches='tight',dpi = 400)

from scipy import stats
print(stats.ttest_rel(Vivo_ANN_DS3_ESM["value"], Vivo_ANN_DS3_OHE["value"])[1])
print(stats.ttest_rel(Vivo_ANN_DS3_ESM["value"], Vivo_ANN_DS3_Zscale["value"])[1])
print(stats.ttest_rel(Vivo_ANN_DS3_OHE["value"], Vivo_ANN_DS3_Zscale["value"])[1])


# # Time-line of projects (Figure)

# In[42]:


import plotly.express as px
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
## Load data from output of "get_data.ipynb"
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
# Add time signature of project.
protocol = pd.read_excel("../data/raw/protocol_results_ny.xlsx")
protocol.set_index("NNCNo API",inplace=True)
protocol.index.names = ["nncno"]
protocol_date = protocol["UploadDate"]
PK_data_mean_up = pd.read_excel("../data/raw/nonSedatedRatData_Updated.xlsx")
PK_data_mean_up.set_index("analouge",inplace=True)
PK_data_mean_up.index.names = ["nncno"]
PK_mean_data_with_dates = pd.merge(PK_data_mean_up, protocol_date, left_index=True, right_index=True).reset_index().groupby("nncno").first()
dates = PK_mean_data_with_dates["UploadDate"]

Full_data_with_time = pd.merge(Full_data,dates,left_index=True, right_index=True)
Full_data_with_time["UploadDate"] = pd.to_datetime(Full_data_with_time["UploadDate"])
Full_data_with_time.set_index("UploadDate",inplace=True,append =True)
Full_data_with_time.sort_index(level = "UploadDate",inplace=True)
Full_data_with_time["Project"] = Full_data_with_time.index.get_level_values("nncno").str.slice(0,4)
Start_end_dates = Full_data_with_time.reset_index().groupby(['Project'], as_index=False).agg({'UploadDate' : [min, max]})
Start_end_dates.columns = ["Project","Start_date","End_date"]
Start_end_dates["Index"] = range(0,16)
Start_end_dates["Start_date"] = pd.to_datetime(Start_end_dates["Start_date"],utc=True)
Start_end_dates["End_date"] = pd.to_datetime(Start_end_dates["End_date"],utc=True)
# Just to show the bar instead of having blank space:
Start_end_dates["End_date"] = np.where(Start_end_dates["Project"] == "0403" , Start_end_dates["End_date"] + timedelta(days=20), Start_end_dates["End_date"])
Start_end_dates["End_date"] = np.where(Start_end_dates["Project"] == "0276" , Start_end_dates["End_date"] + timedelta(days=20), Start_end_dates["End_date"])
Start_end_dates["Project"] = pd.Series(range(0,16)).astype("str")


KFold_outer = TimeSeriesSplit(n_splits = 5)
mins = pd.DataFrame()
for j, (outer_train_idx, test_idx) in enumerate(KFold_outer.split(Full_data_with_time, groups = Full_data_with_time.index.get_level_values("UploadDate"))):
        data_train_outer, data_test = Full_data_with_time.iloc[outer_train_idx], Full_data_with_time.iloc[test_idx]
        min_ = data_test.reset_index()["UploadDate"].min()
        max_ = data_test.reset_index()["UploadDate"].max()
        #print(max_)
        
        #print(max_)


# In[43]:


## pass an explicit array to the color_discrete_sequence 
plt.figure(figsize=(10,10))
fig = px.timeline(Start_end_dates, x_start="Start_date", x_end="End_date", y="Project", 
    color="Project", color_discrete_sequence=Start_end_dates["Index"].values,width=800, height=500)
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)
fig.add_vline(x= "2013-03-20 00:00:00+01:00")
fig.add_vline(x= "2014-06-13 00:00:00+02:00")
fig.add_vline(x= "2016-03-09 00:00:00+01:00")
fig.add_vline(x= "2017-02-01 00:00:00+01:00")
fig.add_vline(x= "2017-12-18 00:00:00+01:00")

fig.add_annotation(x="2013-11-20 00:00:00+01:00", y=16,
            showarrow=False,
            text="Test1")
fig.add_annotation(x="2015-04-13 00:00:00+02:00", y=16,
            showarrow=False,
            text="Test2")
fig.add_annotation(x="2016-08-14 00:00:00+02:00", y=16,
            showarrow=False,
            text="Test3")
fig.add_annotation(x="2017-06-30 00:00:00+02:00", y=16,
            showarrow=False,
            text="Test4")
fig.add_annotation(x="2019-06-30 00:00:00+02:00", y=16,
            showarrow=False,
            text="Test5")
fig.show()



# # Mean predictor plots

# In[44]:


RF_MP = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_MeanPredictor_original.pkl','rb')) 
RF_MP_ts = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_ts_MeanPredictor_original.pkl','rb'))
RF_MP_ts["DS1"]
dict_save = {"Random split":RF_MP["DS1"],"Temporal split":RF_MP_ts["DS1"]}
pd_save = pd.DataFrame(dict_save).melt()

plt.style.use('ggplot')
sns.set(font_scale=2)
g = sns.catplot(x="variable", y="value", data=pd_save, kind="box",palette=sns.color_palette("muted"),height=7)
g.map(sns.swarmplot, 'variable', 'value', color='k')
g.set(xlabel='Split ', ylabel='RMSE (CL)',title = "",yticks = np.arange(2,10,1))
g.set(ylim=(0, 10))
#[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]

g.savefig('../reports/figures/Mean_predictor_plots.png', bbox_inches='tight') 


# In[45]:


RF_MP = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_MeanPredictor.pkl','rb'))
RF_MP_ts = pickle.load(open('/home/kyei/insulin_pk_prediction/data/processed/Random_forest_outer_test_scorer_ts_MeanPredictor.pkl','rb'))


dict_save = {"Random split":RF_MP["DS1"],"Temporal split":RF_MP_ts["DS1"]}
pd_save = pd.DataFrame(dict_save).melt()

plt.style.use('ggplot')
sns.set(font_scale=2)
g = sns.catplot(x="variable", y="value", data=pd_save, kind="box",palette=sns.color_palette("muted"),height=7)
g.map(sns.swarmplot, 'variable', 'value', color='k')
g.set(xlabel='Split ', ylabel='RMSE (CL)',title = "",yticks = np.arange(1,1.7,0.1))
g.set(ylim=(1, 1.7))
#[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]

g.savefig('../reports/figures/Mean_predictor_plots.png', bbox_inches='tight') 


# # Fold errors

# In[46]:


def calculate_FE(data):
    
    data["pred_abs"] = np.abs(data["predictions"])
    data["obs_abs"] = np.abs(data["observations"])
    
    data["fold_error"] = np.where(data["obs_abs"] > data["pred_abs"], data["obs_abs"]/data["pred_abs"], data["pred_abs"]/data["obs_abs"])
    data["fold_error_log"] = np.log(data["fold_error"])
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    
    percent_2 = data[data["fold_error"] < 2].shape[0]/data.shape[0]*100
    percent_3 = data[data["fold_error"] < 3].shape[0]/data.shape[0]*100
    percent_5 = data[data["fold_error"] < 5].shape[0]/data.shape[0]*100
    
    
    AFE = 10**((1/data.shape[0])*np.sum(data["fold_error_log"]))
    
    return [percent_2,percent_3,percent_5, AFE]
    
    


# In[47]:


Random_forest_scatter = pickle.load(open('../data/processed/Random_forest_scatter_file.pkl','rb'))
RF_124 = Random_forest_scatter["DS124"]
RF_124_wide = RF_124.pivot(index = ["nncno","variable"],columns = "Type", values = "value").reset_index()

RF_124_FE_data_CL = calculate_FE(RF_124_wide[RF_124_wide["variable"] == "CL[ml/min/kg]"])
RF_124_FE_data_MRT = calculate_FE(RF_124_wide[RF_124_wide["variable"] == "MRT[h]"])
RF_124_FE_data_T12 = calculate_FE(RF_124_wide[RF_124_wide["variable"] == "T1/2[h]"])


# In[ ]:


# FE temporal splits

Random_forest_scatter_ts = pickle.load(open('../data/processed/Random_forest_scatter_file_ts.pkl','rb'))
RF_124_ts = Random_forest_scatter_ts["DS124"]
RF_124_wide_ts = RF_124_ts.pivot(index = ["nncno","variable"],columns = "Type", values = "value").reset_index()

RF_124_FE_data_CL_ts = calculate_FE(RF_124_wide_ts[RF_124_wide_ts["variable"] == "CL[ml/min/kg]"])
RF_124_FE_data_MRT_ts = calculate_FE(RF_124_wide_ts[RF_124_wide_ts["variable"] == "MRT[h]"])
RF_124_FE_data_T12_ts = calculate_FE(RF_124_wide_ts[RF_124_wide_ts["variable"] == "T1/2[h]"])


# In[ ]:


# Random splits
scatter_1234_ANN = extract_ANN_data_for_plot(ANN_total_data,"DS1234")
ANN_1234_wide = scatter_1234_ANN.pivot(index = ["nncno","variable"],columns = "Type", values = "value").reset_index()

ANN_1234_FE_data_CL = calculate_FE(ANN_1234_wide[ANN_1234_wide["variable"] == "CL[ml/min/kg]"])
ANN_1234_FE_data_MRT = calculate_FE(ANN_1234_wide[ANN_1234_wide["variable"] == "MRT[h]"])
ANN_1234_FE_data_T12 = calculate_FE(ANN_1234_wide[ANN_1234_wide["variable"] == "T1/2[h]"])



# In[ ]:


scatter_1234_ANN_ts = extract_ANN_data_for_plot_ts(ANN_total_data_ts,"DS1234")
ANN_1234_wide_ts = scatter_1234_ANN_ts.pivot(index = ["nncno","variable"],columns = "Type", values = "value").reset_index()

ANN_1234_FE_data_CL_ts = calculate_FE(ANN_1234_wide_ts[ANN_1234_wide_ts["variable"] == "CL[ml/min/kg]"])
ANN_1234_FE_data_MRT_ts = calculate_FE(ANN_1234_wide_ts[ANN_1234_wide_ts["variable"] == "MRT[h]"])
ANN_1234_FE_data_T12_ts = calculate_FE(ANN_1234_wide_ts[ANN_1234_wide_ts["variable"] == "T1/2[h]"])


# # Experimental error (best achievable model)

# In[ ]:


data_with_std = pd.read_excel("../data/raw/In_vivo_rat_IV.xlsx")
data_with_std.columns.values[0] = "nncno"
data_with_std.set_index("nncno",inplace=True)


# In[ ]:


data_with_std = data_with_std[["[SD.(CL).(ml/min/kg)]","[SD.T1/2.(h)]", "[SD.MRT.(h)]"]]
PK_data = Full_data[['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']]


# In[ ]:


data_to = pd.merge(PK_data, data_with_std, left_index=True, right_index=True).reset_index()


# # Data for table 1

# In[ ]:


Data_with_groups = pd.read_excel("../data/raw/Data_with_groups.xlsx")
Data_with_groups.rename(columns={"NNCNo": "nncno"})
Data_with_groups.set_index("NNCNo",inplace=True)
Data_with_groups = Data_with_groups[~Data_with_groups.index.isin(["0148-0000-1247"])]
Full_data = pd.read_csv("../data/processed/full_data_set.csv")
Full_data.set_index("nncno",inplace=True)
PK_names = ["CL[ml/min/kg]","T1/2[h]","MRT[h]"]
PK_SD_names = ["[SD.(CL).(ml/min/kg)]", "[SD.T1/2.(h)]", "[SD.MRT.(h)]"]


# In[ ]:


Data_with_groups["Groups"].value_counts()


# In[ ]:


data_to_no_na = data_to.dropna()


# In[ ]:


data_to_no_na["ln_sd_cl"] =  data_to_no_na["[SD.(CL).(ml/min/kg)]"]/data_to_no_na["CL[ml/min/kg]"]
data_to_no_na["ln_sd_T12"] =  np.abs(data_to_no_na["[SD.T1/2.(h)]"]/data_to_no_na["T1/2[h]"])
data_to_no_na["ln_sd_MRT"] =  np.abs(data_to_no_na["[SD.MRT.(h)]"]/data_to_no_na["MRT[h]"])

data_to_no_na["ln_var_cl"] =  (data_to_no_na["[SD.(CL).(ml/min/kg)]"]/data_to_no_na["CL[ml/min/kg]"])**2
data_to_no_na["ln_var_T12"] =  (np.abs(data_to_no_na["[SD.T1/2.(h)]"]/data_to_no_na["T1/2[h]"]))**2
data_to_no_na["ln_var_MRT"] =  (np.abs(data_to_no_na["[SD.MRT.(h)]"]/data_to_no_na["MRT[h]"]))**2

data_grouped_nncno_no_na = data_to_no_na.groupby("nncno").mean()


# In[ ]:


data_to["ln_sd_cl"] =  data_to["[SD.(CL).(ml/min/kg)]"]/data_to["CL[ml/min/kg]"]
data_to["ln_sd_T12"] =  np.abs(data_to["[SD.T1/2.(h)]"]/data_to["T1/2[h]"])
data_to["ln_sd_MRT"] =  np.abs(data_to["[SD.MRT.(h)]"]/data_to["MRT[h]"])

data_to["ln_var_cl"] =  (data_to["[SD.(CL).(ml/min/kg)]"]/data_to["CL[ml/min/kg]"])**2
data_to["ln_var_T12"] =  (np.abs(data_to["[SD.T1/2.(h)]"]/data_to["T1/2[h]"]))**2
data_to["ln_var_MRT"] =  (np.abs(data_to["[SD.MRT.(h)]"]/data_to["MRT[h]"]))**2

data_grouped_nncno = data_to.groupby("nncno").mean()


# In[ ]:


mean_sd_CL = np.sqrt(data_grouped_nncno["ln_var_cl"].mean())
mean_sd_T12 = np.sqrt(data_grouped_nncno["ln_var_T12"].mean())
mean_sd_MRT = data_grouped_nncno["ln_var_MRT"].mean()


# In[ ]:


sample_sd_CL = np.sqrt((np.sum((data_grouped_nncno["ln_sd_cl"] - mean_sd_CL)**2))/(data_grouped_nncno.shape[0]-1))
sample_sd_T12 = np.sqrt((np.sum((data_grouped_nncno["ln_sd_T12"] - mean_sd_T12)**2))/(data_grouped_nncno.shape[0]-1))
sample_sd_MRT = np.sqrt((np.sum((data_grouped_nncno["ln_sd_MRT"] - mean_sd_MRT)**2))/(data_grouped_nncno.shape[0]-1))




# In[ ]:


print(np.sqrt(data_to["ln_var_cl"].mean()))
print(np.sqrt(data_to["ln_var_T12"].mean()))
np.sqrt(data_to["ln_var_MRT"].mean())


# In[ ]:


print(sample_sd_CL)
print(sample_sd_T12)
print(sample_sd_MRT)


# In[ ]:


print(mean_sd_CL+1.96*sample_sd_CL)
print(mean_sd_CL-1.96*sample_sd_CL)


print(mean_sd_T12+1.96*sample_sd_T12)
print(mean_sd_T12-1.96*sample_sd_T12)


print(mean_sd_MRT+1.96*sample_sd_MRT)
print(mean_sd_MRT-1.96*sample_sd_MRT)


# In[ ]:


g= sns.boxplot(data = data_grouped_nncno[["ln_sd_cl","ln_sd_T12", "ln_sd_MRT"]].melt(),x = "variable",y = "value")
g.set(ylim=(-0.05, 0.5))


# In[ ]:


pd.DataFrame(data_grouped_nncno


# In[ ]:


data_wg = pd.merge(data_to, Data_with_groups["Groups"], left_index=True, right_index=True)


# In[ ]:


data_wg.groupby(["Groups"])[PK_names].min()


# In[ ]:


data_wg.groupby(["Groups"])[PK_names].max()


# In[ ]:


data_wg.groupby(["Groups"])[PK_SD_names].mean()


# In[ ]:


data_wg["[SD.(CL).(ml/min/kg)]"].mean()


# # End of results
