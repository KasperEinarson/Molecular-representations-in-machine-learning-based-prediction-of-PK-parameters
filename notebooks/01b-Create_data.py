#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Retrieve data from novo nordisk internal database both on descriptors and PK data.
Select descriptors of interest and create embeddings from sequential data.
Output: Exports final (full) dataset for further analysis.

'''
import novopy
from novodataset.dataset import SAR4MLDataSet
import pandas as pd
import numpy as np
import random
from bio_embeddings.embed import ESM1bEmbedder
from sklearn.preprocessing import OrdinalEncoder
import sys
import os
from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from novodataset.dataset import SAR4MLDataSet
w2v_model = word2vec.Word2Vec.load('../models/model_mol2vec.pkl')
from ast import literal_eval

# Set seed
seed = 42
np.random.seed(seed)
random.seed(seed)

get_ipython().system('jupyter nbconvert --to script "01b-Create_data.ipynb"')


# In[2]:


'''
Functions needed for importing and selecting data appropriately for all descriptorsets + PK data.

'''

def select_descriptors_and_analogs(path = "../data/processed/Descriptors_directly_from_nncd.csv"):
    '''
    Function to load descriptors from csv output from "01a-insilico_descriptors.py" and return in format used for this analysis.
    input: Path to descriptor csv file
    output: Full_data - all descriptorsets for all insulin analogs used in this analysis.
    
    '''
    
    # Descriptors_directly_from_nncd.csv has been created by running python 01a-insilico_descriptors.py. Read SMILES as list instead of default strings:
    Descriptors = pd.read_csv(path,converters={"insilico2d:protractor:smiles": literal_eval} )
    Descriptors.set_index("nncno",inplace=True)
    Descriptors.drop("Unnamed: 0",axis=1,inplace=True)
    # DS1:
    DS1_Descriptors = Descriptors.drop(Descriptors.columns[pd.Series(Descriptors.columns).str.startswith('insilico2d:protractor')],axis=1)
    DS1_Descriptors.drop("seq",axis=1,inplace=True)
    DS1_Descriptors.columns = DS1_Descriptors.columns.str.replace('^insilico2d:', 'DS1_')
    # DS2:
    DS2_Descriptors = Descriptors[Descriptors.columns[pd.Series(Descriptors.columns).str.startswith('insilico2d:protractor')]]
    DS2_Descriptors.drop("insilico2d:protractor:smiles",axis=1,inplace=True)
    DS2_Descriptors.columns = DS2_Descriptors.columns.str.replace('^insilico2d:protractor:', 'DS2_')
    # DS3 (raw sequences):
    DS3_raw = Descriptors["seq"]
    # DS4 (raw SMILES):
    DS4_raw = Descriptors['insilico2d:protractor:smiles']
    ## Small name adjustments:
    DS1_Descriptors = DS1_Descriptors.rename({'ALOGP_PP': 'DS1_ALOGP_PP', 'PEP_BOND_COUNT': 'DS1_PEP_BOND_COUNT'}, axis=1) 
    
    
    
    return DS1_Descriptors, DS2_Descriptors, DS3_raw, DS4_raw


def load_PK_data(path):
    '''
    Load PK data, set index and remove invalid PK observations.
    '''
    
    
    PK = pd.read_excel(path)
    PK.rename(columns={"analouge":"nncno"},inplace = True)
    PK.set_index("nncno",inplace=True)
    PK = PK[PK['CL[ml/min/kg]'].notna()]
    PK.drop("Vz[ml/kg]",axis=1,inplace=True)
    return PK


def make_DS3_ESM_embedding(DS3_raw, save_path = "../data/processed/pandas_ESM1b_Vivo.csv"):
    '''
    Use ESM 1b embedding from bio embeddings module in python to create protein embedding
    input: Raw amino acid sequence (un-aligned)
    output: ESM1b embedding with 1280 dimensions.
    '''
    
    ESM1b = ESM1bEmbedder()
    ESM1b_embeddings = ESM1b.embed_many(DS3_raw)
    ESM1b_trans = [ESM1bEmbedder.reduce_per_protein(e) for e in ESM1b_embeddings]
    ESM1b_trans_pd = pd.DataFrame(ESM1b_trans)
    ESM1b_trans_pd["nncno"] = DS3_raw.index
    ESM1b_trans_pd.to_csv(save_path)
    return ESM1b_trans_pd


def remove_None(SMILES_string):
    '''
    Some SMILES strings contains "None" which counts as an acylation (which is not correct..)
    We therefore remove any None entries in the SMILES input
    '''
    DS4_raw_no_none = []
    for i in range(DS4_raw.shape[0]):
        DS4_raw_no_none.append([x for x in SMILES_string[i] if x is not None])
    DS4_raw_no_none = pd.Series(DS4_raw_no_none,index = SMILES_string.index)
    return DS4_raw_no_none
    

def make_DS4_embedding(SMILES_string, Concat = "sum"):
    '''
    Apologies for the hardcoding.
    Function to calculate SMILES embedding of small molecule part. 
    In case of multiple protractors, the sum of the embeddings is calculated
    Input: SMILES string of protractor
    Input: Concat = How should multiple protractor embeddings be concatenated? options are "mean" or "sum".
    output: DataFrame with embedding n x d, n = number of smiles string, d = embedding dimension. 
    
    '''
    SMILES_string_divided = SMILES_string.apply(lambda d: d if isinstance(d, list) else [])
    # How many protracto?
    lenght_prot = []
    for i in range(len(SMILES_string_divided)):
        lenght_prot.append(len(SMILES_string_divided[i]))

    length_pd = pd.DataFrame(np.array(lenght_prot),columns = ["Length"],index = SMILES_string_divided.index)
    #print(length_pd.value_counts())
    # Handle 1 protractor 
    Pd_1_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 1].index)][i][0] for i in range(length_pd[length_pd.Length == 1].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 1].index) 
    Pd_1_protractor["mol"] = Pd_1_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_1_protractor['sentence'] = Pd_1_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_1_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_1_protractor['sentence'], w2v_model)]
    Pd_1_protractor_embedding = np.array([x.vec for x in Pd_1_protractor['embedding']])
    Pd_1_protractor_embedding = pd.DataFrame(Pd_1_protractor_embedding)
    Pd_1_protractor_embedding.index = length_pd[length_pd.Length == 1].index
    Pd_1_protractor_embedding = Pd_1_protractor_embedding.add_prefix('SMILES_')
    # Handle zero protractors (easy..)
    Pd_0_protractors = pd.DataFrame(np.zeros((length_pd[length_pd.Length == 0].shape[0], 100)),index = length_pd[length_pd.Length == 0].index,columns = Pd_1_protractor_embedding.columns)

    # Handle 2 protractors
    Pd_21_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 2].index)][i][0] for i in range(length_pd[length_pd.Length == 2].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 2].index) 
    Pd_22_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 2].index)][i][1] for i in range(length_pd[length_pd.Length == 2].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 2].index) 

    Pd_21_protractor["mol"] = Pd_21_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_22_protractor["mol"] = Pd_22_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

    Pd_21_protractor['sentence'] = Pd_21_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_22_protractor['sentence'] = Pd_22_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)

    Pd_21_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_21_protractor['sentence'], w2v_model)]
    Pd_22_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_22_protractor['sentence'], w2v_model)]

    Pd_21_protractor_embedding = np.array([x.vec for x in Pd_21_protractor['embedding']])
    Pd_22_protractor_embedding = np.array([x.vec for x in Pd_22_protractor['embedding']])

    Pd_21_protractor_embedding = pd.DataFrame(Pd_21_protractor_embedding)
    Pd_22_protractor_embedding = pd.DataFrame(Pd_22_protractor_embedding)

    Pd_21_protractor_embedding.index = length_pd[length_pd.Length == 2].index
    Pd_22_protractor_embedding.index = length_pd[length_pd.Length == 2].index

    Pd_21_protractor_embedding = Pd_21_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_22_protractor_embedding = Pd_22_protractor_embedding.add_prefix('SMILES_').sort_index()


    # Handle 3 protractors
    Pd_31_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 3].index)][i][0] for i in range(length_pd[length_pd.Length == 3].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 3].index) 
    Pd_32_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 3].index)][i][1] for i in range(length_pd[length_pd.Length == 3].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 3].index) 
    Pd_33_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 3].index)][i][2] for i in range(length_pd[length_pd.Length == 3].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 3].index) 
 
    Pd_31_protractor["mol"] = Pd_31_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_32_protractor["mol"] = Pd_32_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_33_protractor["mol"] = Pd_33_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

    Pd_31_protractor['sentence'] = Pd_31_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_32_protractor['sentence'] = Pd_32_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_33_protractor['sentence'] = Pd_33_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)

    Pd_31_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_31_protractor['sentence'], w2v_model)]
    Pd_32_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_32_protractor['sentence'], w2v_model)]
    Pd_33_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_33_protractor['sentence'], w2v_model)]

    Pd_31_protractor_embedding = np.array([x.vec for x in Pd_31_protractor['embedding']])
    Pd_32_protractor_embedding = np.array([x.vec for x in Pd_32_protractor['embedding']])
    Pd_33_protractor_embedding = np.array([x.vec for x in Pd_33_protractor['embedding']])

    Pd_31_protractor_embedding = pd.DataFrame(Pd_31_protractor_embedding)
    Pd_32_protractor_embedding = pd.DataFrame(Pd_32_protractor_embedding)
    Pd_33_protractor_embedding = pd.DataFrame(Pd_33_protractor_embedding)

    Pd_31_protractor_embedding.index = length_pd[length_pd.Length == 3].index
    Pd_32_protractor_embedding.index = length_pd[length_pd.Length == 3].index
    Pd_33_protractor_embedding.index = length_pd[length_pd.Length == 3].index

    Pd_31_protractor_embedding = Pd_31_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_32_protractor_embedding = Pd_32_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_33_protractor_embedding = Pd_33_protractor_embedding.add_prefix('SMILES_').sort_index()

    
    # Handle 4 protractors:
    
    Pd_41_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 4].index)][i][0] for i in range(length_pd[length_pd.Length == 4].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 4].index) 
    Pd_42_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 4].index)][i][1] for i in range(length_pd[length_pd.Length == 4].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 4].index) 
    Pd_43_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 4].index)][i][2] for i in range(length_pd[length_pd.Length == 4].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 4].index) 
    Pd_44_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 4].index)][i][3] for i in range(length_pd[length_pd.Length == 4].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 4].index) 
 
    Pd_41_protractor["mol"] = Pd_41_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_42_protractor["mol"] = Pd_42_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_43_protractor["mol"] = Pd_43_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_44_protractor["mol"] = Pd_44_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

    Pd_41_protractor['sentence'] = Pd_41_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_42_protractor['sentence'] = Pd_42_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_43_protractor['sentence'] = Pd_43_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_44_protractor['sentence'] = Pd_44_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)

    Pd_41_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_41_protractor['sentence'], w2v_model)]
    Pd_42_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_42_protractor['sentence'], w2v_model)]
    Pd_43_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_43_protractor['sentence'], w2v_model)]
    Pd_44_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_44_protractor['sentence'], w2v_model)]

    
    Pd_41_protractor_embedding = np.array([x.vec for x in Pd_41_protractor['embedding']])
    Pd_42_protractor_embedding = np.array([x.vec for x in Pd_42_protractor['embedding']])
    Pd_43_protractor_embedding = np.array([x.vec for x in Pd_43_protractor['embedding']])
    Pd_44_protractor_embedding = np.array([x.vec for x in Pd_44_protractor['embedding']])

    Pd_41_protractor_embedding = pd.DataFrame(Pd_41_protractor_embedding)
    Pd_42_protractor_embedding = pd.DataFrame(Pd_42_protractor_embedding)
    Pd_43_protractor_embedding = pd.DataFrame(Pd_43_protractor_embedding)
    Pd_44_protractor_embedding = pd.DataFrame(Pd_44_protractor_embedding)

    Pd_41_protractor_embedding.index = length_pd[length_pd.Length == 4].index
    Pd_42_protractor_embedding.index = length_pd[length_pd.Length == 4].index
    Pd_43_protractor_embedding.index = length_pd[length_pd.Length == 4].index
    Pd_44_protractor_embedding.index = length_pd[length_pd.Length == 4].index

    Pd_41_protractor_embedding = Pd_41_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_42_protractor_embedding = Pd_42_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_43_protractor_embedding = Pd_43_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_44_protractor_embedding = Pd_44_protractor_embedding.add_prefix('SMILES_').sort_index()

    
    
     # Handle 7 protractors:
    
    Pd_71_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][0] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_72_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][1] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_73_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][2] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_74_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][3] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_75_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][4] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_76_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][5] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
    Pd_77_protractor = pd.DataFrame([SMILES_string_divided[SMILES_string_divided.index.isin(length_pd[length_pd.Length == 7].index)][i][6] for i in range(length_pd[length_pd.Length == 7].shape[0])],columns = ["SMILES"],index = length_pd[length_pd.Length == 7].index) 
 
    Pd_71_protractor["mol"] = Pd_71_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_72_protractor["mol"] = Pd_72_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_73_protractor["mol"] = Pd_73_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_74_protractor["mol"] = Pd_74_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_75_protractor["mol"] = Pd_75_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_76_protractor["mol"] = Pd_76_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    Pd_77_protractor["mol"] = Pd_77_protractor["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

    Pd_71_protractor['sentence'] = Pd_71_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_72_protractor['sentence'] = Pd_72_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_73_protractor['sentence'] = Pd_73_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_74_protractor['sentence'] = Pd_74_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_75_protractor['sentence'] = Pd_75_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_76_protractor['sentence'] = Pd_76_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)
    Pd_77_protractor['sentence'] = Pd_77_protractor.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], radius=1)), axis=1)

    Pd_71_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_71_protractor['sentence'], w2v_model)]
    Pd_72_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_72_protractor['sentence'], w2v_model)]
    Pd_73_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_73_protractor['sentence'], w2v_model)]
    Pd_74_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_74_protractor['sentence'], w2v_model)]
    Pd_75_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_75_protractor['sentence'], w2v_model)]
    Pd_76_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_76_protractor['sentence'], w2v_model)]
    Pd_77_protractor['embedding'] = [DfVec(x) for x in sentences2vec(Pd_77_protractor['sentence'], w2v_model)]

    
    Pd_71_protractor_embedding = np.array([x.vec for x in Pd_71_protractor['embedding']])
    Pd_72_protractor_embedding = np.array([x.vec for x in Pd_72_protractor['embedding']])
    Pd_73_protractor_embedding = np.array([x.vec for x in Pd_73_protractor['embedding']])
    Pd_74_protractor_embedding = np.array([x.vec for x in Pd_74_protractor['embedding']])
    Pd_75_protractor_embedding = np.array([x.vec for x in Pd_75_protractor['embedding']])
    Pd_76_protractor_embedding = np.array([x.vec for x in Pd_76_protractor['embedding']])
    Pd_77_protractor_embedding = np.array([x.vec for x in Pd_77_protractor['embedding']])

    Pd_71_protractor_embedding = pd.DataFrame(Pd_71_protractor_embedding)
    Pd_72_protractor_embedding = pd.DataFrame(Pd_72_protractor_embedding)
    Pd_73_protractor_embedding = pd.DataFrame(Pd_73_protractor_embedding)
    Pd_74_protractor_embedding = pd.DataFrame(Pd_74_protractor_embedding)
    Pd_75_protractor_embedding = pd.DataFrame(Pd_75_protractor_embedding)
    Pd_76_protractor_embedding = pd.DataFrame(Pd_76_protractor_embedding)
    Pd_77_protractor_embedding = pd.DataFrame(Pd_77_protractor_embedding)

    Pd_71_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_72_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_73_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_74_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_75_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_76_protractor_embedding.index = length_pd[length_pd.Length == 7].index
    Pd_77_protractor_embedding.index = length_pd[length_pd.Length == 7].index

    Pd_71_protractor_embedding = Pd_71_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_72_protractor_embedding = Pd_72_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_73_protractor_embedding = Pd_73_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_74_protractor_embedding = Pd_74_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_75_protractor_embedding = Pd_75_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_76_protractor_embedding = Pd_76_protractor_embedding.add_prefix('SMILES_').sort_index()
    Pd_77_protractor_embedding = Pd_77_protractor_embedding.add_prefix('SMILES_').sort_index()

    
    
    # Concatenation of multiple protractors (mean or sum?)
    if Concat == "sum":
        Pd_2_protractor_embeddingp_mean = pd.concat([Pd_21_protractor_embedding, Pd_22_protractor_embedding]).groupby(level=0).sum()
        Pd_3_protractor_embeddingp_mean = pd.concat([Pd_31_protractor_embedding, Pd_32_protractor_embedding, Pd_33_protractor_embedding]).groupby(level=0).sum()
        Pd_4_protractor_embeddingp_mean = pd.concat([Pd_41_protractor_embedding, Pd_42_protractor_embedding, Pd_43_protractor_embedding,Pd_44_protractor_embedding]).groupby(level=0).sum()
        Pd_7_protractor_embeddingp_mean = pd.concat([Pd_71_protractor_embedding, Pd_72_protractor_embedding, Pd_73_protractor_embedding,Pd_74_protractor_embedding,Pd_75_protractor_embedding,Pd_76_protractor_embedding,Pd_77_protractor_embedding ]).groupby(level=0).sum()
    elif Concat == "mean":
        Pd_2_protractor_embeddingp_mean = pd.concat([Pd_21_protractor_embedding, Pd_22_protractor_embedding]).groupby(level=0).mean()
        Pd_3_protractor_embeddingp_mean = pd.concat([Pd_31_protractor_embedding, Pd_32_protractor_embedding, Pd_33_protractor_embedding]).groupby(level=0).mean()
        Pd_4_protractor_embeddingp_mean = pd.concat([Pd_41_protractor_embedding, Pd_42_protractor_embedding, Pd_43_protractor_embedding,Pd_44_protractor_embedding]).groupby(level=0).mean()
        Pd_7_protractor_embeddingp_mean = pd.concat([Pd_71_protractor_embedding, Pd_72_protractor_embedding, Pd_73_protractor_embedding,Pd_74_protractor_embedding,Pd_75_protractor_embedding,Pd_76_protractor_embedding,Pd_77_protractor_embedding ]).groupby(level=0).mean()
    
    
    
    
    # Concatenate all data together:
    SMILES_word2vec_embedding = pd.concat([Pd_0_protractors,Pd_1_protractor_embedding, Pd_2_protractor_embeddingp_mean,Pd_3_protractor_embeddingp_mean,Pd_4_protractor_embeddingp_mean,Pd_7_protractor_embeddingp_mean])
    
    
    
    
    return SMILES_word2vec_embedding


# # Load PK data and descriptors for all analogs using above functions

# In[3]:


DS1, DS2, DS3_raw, DS4_raw = select_descriptors_and_analogs(path = "../data/processed/Descriptors_directly_from_nncd.csv")
PK_data = load_PK_data("../data/raw/nonSedatedRatData_Updated.xlsx")
PK_data = PK_data[PK_data.index.isin(DS1.index)]
DS4_raw_no_none = remove_None(DS4_raw)


# In[4]:


# Make DS3 embedding:
# Takes DS3_raw and produces the embedding. Following line of code takes ~30 minutes to run on standard laptop and is therefore saved as a seperate data-file once processed.
#DS3 = make_DS3_ESM_embedding(DS3_raw,save_path = "../data/processed/pandas_ESM1b_Vivo.csv")
DS3 = pd.read_csv("../data/processed/pandas_ESM1b_Vivo.csv")
DS3.drop("Unnamed: 0",axis=1,inplace=True)
DS3.set_index("nncno",inplace=True)                  
DS3 = DS3.add_prefix('DS3_')

# Make DS4 embedding:
DS4 = make_DS4_embedding(DS4_raw_no_none,Concat = "sum")
DS4.columns = DS4.columns.str.replace('^SMILES', 'DS4')


# In[5]:


# Export full dataset (all descriptors + PK data)
Full_data = pd.concat([DS1,DS2,DS3,DS4,PK_data],axis=1)
# Last row is all NAs:
Full_data = Full_data[:-1]


# In[6]:


# Save the full data set to csv:
Full_data.to_csv("../data/processed/full_data_set.csv")

