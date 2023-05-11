# PK prediction of insulin analogs using combination of molecular descriptors

## Analysis divided into 5 parts indicated by prefix in notebook and python files
1. Extract and create dataset (A: Create descriptors from nncd number (can only run if you have permission to all nncd), B: Create descriptors for analysis)
2. Split data into folds (only used for ANN) - grouped data as well
3. Modelling (A: Random Forest, B: ANN) (A2 for grouped data analysis) 
4. Calculation of SHAP values
5. Figures and results with additional experiments carried out.
6. Supporting Information (A: Group-calculation used for Table 3 and S5, B: Figures/tables for Supporting information and table 3)

## Following molecular descriptors are considered for modelling:
1. DS1 (full molecule descriptors): alogp, pep_bond_count, length, hydrophobic, ch_acid, charge, charge_density,
       hydrophobic_ratio,non_polar_ratio, aliphatic_ratio,
       boman_index, flexibility, pi, mw, ss_bridge_count
2. DS2 (numeric descriptors of protractor): num, molwt, numrotatablebonds, mollogp,
       maxestateindex', minestateindex', tpsa
3. DS3: ESM1b-embedding of (back-bone) amino acid sequence in single letters
4. DS4: mol2vec of SMILES representation of protractor.

## More details on each analysis step

1. **Create_data:** Retrieve data from novo nordisk internal database both on descriptors and PK data.
    Select descriptors of interest and create embeddings from sequential data.
    Output: Exports final (full) dataset for further analysis. Name: full_data_set.csv and is saved in /data/processed/. 
    ESM embedding (DS3) is made using the *bio embeddings* module in python in the function *make_DS3_ESM_embedding*  
    Word2vec embeddings (DS4) from smiles is made using *gensim* module in python in the function *make_DS4_embedding*
    
2. **Split_data:** Notebook to split data into a user defined amount of folds for training, validation and test. This is done for the full descriptorset (DS1234) such that subsets of descriptors can be extracted in the model notebooks (RF, ANN).
    Splitting the data here is only used for ANN model later on. The exact same splitting is done within "Random_Forest.ipynb/py" for RF model seperately.
    Output: List of dataframes exported to folder "Processed data" which contains all relevant data splitted and scaled for each fold and ready as model (ANN) input.
    Splitting data for each insulin group is also carried out here for table S5 (global/local insulin group training)

3.  **ANN_Numeric:** FeedForward neural network on all numeric descriptor combinations: DS12, DS1, DS2  
    **ANN_DS3models:** Feedforward and 1 CNN (with 2 layers) to model combinations with numeric and AA seq (DS3): DS123,DS13,DS3,DS23  
    **ANN_DS4models:** Feedforward and 1 CNN (with 2 layers)to model  combinations with numeric and SMILES seq (DS4): DS124,DS14,DS4,DS24  
    **ANN_DS34models:** Feedforward and 2 CNN (both with 2 layers) to model combinations with numeric, AA seq (DS3) and SMILES (DS4) : DS1234,DS134,DS34,DS234  
    
4. **SHAP values** of best ANN model with median hyperparameters from the 5 fold CV. Shap values for RF are calculated within "03A-Random_forest.py". 

5. **Figures** for article saved in "reports -> Figures" with names in accordance with figure names in article.
6. **Supporting Information** for article supporting information saved in "reports -> Figures with names in accordance with figure names in article.
    

