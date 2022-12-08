Insulin_PK_prediction
==============================

Project for predicting insulin PK parameters with code for internal use at Novo Nordisk. 
Python notebooks and scripts are found in "notebooks" **see Readme within "notebook" folder for precise explanation of scripts and how to run them.** 
Main data pkl file is "full_data_set.csv" and can be found in /data/processed/

Overview of modeling flow from data representation of 640 insulin analogs to model evaluation of the two models Random Forest (RF) and Artificial Neural Networks (ANN):

![Analog_example](reports/figures/Figure2.jpg)

The best data representation for each model is compared with baseline representation (DS1) for both Random Forest and ANN.
Here results for PK parameter Clearence (CL) with 4 insulin analogs with public iv rat PK data available highlighted:
![Figure5](reports/figures/Figure5_CL_scatter-Copy1.png)

Clearly showing that including comprehensive molecular descriptors result in better predicting on testset. 

--------
