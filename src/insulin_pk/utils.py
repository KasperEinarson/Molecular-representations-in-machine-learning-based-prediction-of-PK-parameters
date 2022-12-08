from sklearn.metrics import mean_squared_error as mse
import sys, os
import pickle  
import torch
import optuna
import numpy as np
import math
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

loss_fc = torch.nn.MSELoss()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Table of content:


Functions related to data preprocessing:
 -- contruct_descriptor_sets


Functions related to ANN DS1,DS12,DS2:

 -- reset_weights
 -- init_weights
 -- Dataset_FFNN
 -- build_model_custom
 -- build_model_DS1
 -- test_FFNN
 -- train_and_validate_FFNN
 -- objective_DS1
 
Functions related to ANN DS13,DS123,DS23, DS3, DS4,DS124,DS24:
 -- model_DS123
 -- model_DS123_build
 -- train_and_validate_1CNN
 -- test_1CNN
 -- objective_DS123
 -- Dataset_seq_embeddings
 

Functions related to ANN DS34,DS134,DS1234,DS234:
 -- model_DS1234
 -- model_DS1234_build
 -- train_and_validate
 -- dataaset_all_conc
 -- objective_DS1234
 -- test_FFNN
 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''







'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Functions related to data preprocessing

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


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
    
    data_final = pd.DataFrame()
    if DS_set.str.contains("1").values[0]:
        data_final = pd.concat([data_final,DS1],axis=1)
    if DS_set.str.contains("2").values[0]:
        data_final = pd.concat([data_final,DS2],axis=1)
    if DS_set.str.contains("3").values[0]:
        data_final = pd.concat([data_final,DS3],axis=1)  
    if DS_set.str.contains("4").values[0]:
        data_final = pd.concat([data_final,DS4],axis=1)  
    
    if with_PK:
        PK_names = ['CL[ml/min/kg]', 'T1/2[h]', 'MRT[h]']
        PK_data = Full_dataset[PK_names]
        data_final = pd.concat([data_final,PK_data],axis=1) 
    
    #data_final = pd.concat([data_final,PK_data],axis=1)    
    return data_final  













'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Functions related to ANN DS1, DS12, DS2

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def reset_weights(m):
  
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)


class Dataset_FFNN(Dataset):
    '''
    Pytorch dataset class for simple numeric data
    '''
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_out = self.X.iloc[idx]
        Y_out = self.Y.iloc[idx]     
        return torch.tensor(X_out.values),torch.tensor(Y_out.values)

    

def build_model_custom(trial,in_features):
    '''
    Build pytorch FFNN with size depending on hyperparameter inputs (Optuna) 
    Input: 
        - trial: Optuna object that contains access to all hyperparameters
        - in_features: Input features dimension
    Output:
        - Pytorch model from given hyperparameters
    '''
    
    
    layers = []
    n_layers = trial.suggest_int("n_layers", 1,2)
        
    for i in range(n_layers):
        
            out_features = trial.suggest_int("n_units_l{}".format(i), 1, 30)
            dropout = trial.suggest_uniform("dropout_l{}".format(i), 0, 0.6)
        
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            in_features = out_features
        
    layers.append(nn.Linear(in_features, 3))
    
    return nn.Sequential(*layers)



def build_model_DS1(params,in_features):
    '''
    Build the pytorch model (again) from the best hyperparameter setting - independent of optuna "trials" object.
    Input:
        - params: Dict of best hyperparameters found from Optuna.
        - in_feautres: Number of input features
    Output: 
        - Pytorch model from given hyperparameters
    
    '''
    layers = []
    if params["n_layers"] == 1:
        
        layers.append(nn.Linear(in_features, params["n_units_l0"]))
        layers.append(nn.BatchNorm1d(params["n_units_l0"]))
        layers.append(nn.Dropout(params["dropout_l0"]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(params["n_units_l0"], 3))
    
    if params["n_layers"] == 2:
        layers.append(nn.Linear(in_features, params["n_units_l0"]))
        layers.append(nn.BatchNorm1d(params["n_units_l0"]))
        layers.append(nn.Dropout(params["dropout_l0"]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(params["n_units_l0"], params["n_units_l1"]))
        layers.append(nn.BatchNorm1d(params["n_units_l1"]))
        layers.append(nn.Dropout(params["dropout_l1"]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(params["n_units_l1"], 3))
        
    net = nn.Sequential(*layers)
    return net


def train_and_validate_FFNN(Params,Model,Data_train,Data_Val,scaler_Y,EPOCHS,save_model,save_path):
    '''
    Train and Validate function
    
    '''
    
    
    # Initiate dataloaders with batch_size as hyperparameter:
    train_loader = DataLoader(dataset = Data_train,batch_size=Params["Batch_Size"],shuffle=True,num_workers=0,drop_last = True)
    val_loader = DataLoader(dataset = Data_Val,batch_size=Params["Batch_Size"],shuffle=True,drop_last = True)
    train_loss_save = []
    val_loss_save = []
    Model.train(True)
    best_loss = 10000
    # Initialize optimizer
    optimizer = torch.optim.Adam(Model.parameters(), lr=Params['lr'],weight_decay= Params['wd'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(0, EPOCHS):
        # Set current loss value
        train_loss = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, targets = data            
            
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            Model.double()
            outputs_scaled = Model(inputs)
            # Compute loss
            loss = loss_fc(outputs_scaled, targets)    
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # conversion back before reporting loss
            output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
            target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
            train_loss += loss_fc(output_tensor,target_tensor).item()
            
        #Done with training. Now test(validate):    
        Model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # Get inputs
                inputs, targets = data
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                Model.double()
                outputs_scaled = Model(inputs)
                # Compute loss
                loss = loss_fc(outputs_scaled, targets)    
                #Scaled back-loss
                output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
                target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
                test_loss += loss_fc(output_tensor,target_tensor).item()
                
        # Print
        train_loss /= train_loader.__len__()
        test_loss /= val_loader.__len__()
        #scheduler.step(test_loss)
        train_loss = np.sqrt(train_loss)
        test_loss = np.sqrt(test_loss)
        
        train_loss_save.append(train_loss)
        val_loss_save.append(test_loss)
        
        #if epoch % 40 == 0:
        #    print('Epoch %i: Train RMSE: %.3f Val RMSE: %.3f' % (epoch, train_loss, test_loss))
            
        if test_loss < best_loss:
            if save_model:
                torch.save(Model.state_dict(), save_path)
            best_loss = test_loss
            
    return {"best_loss": best_loss, "train_loss": train_loss_save, "val_loss": val_loss_save}

def objective_DS1(trial,Data_train,Data_Val,Scaler_Y,epoch,save_model,save_path):
    '''
    Hyperparameter wrapper function for optuna
    
    '''
    
    Params = {"lr":trial.suggest_loguniform("lr",1e-6,1e-2),
              "Batch_Size":trial.suggest_int("Batch_Size", 12, 28),
              "wd":trial.suggest_uniform("wd",0,1e-1)
             }
    in_features = next(iter(DataLoader(Data_train)))[0].shape[1]
    Model = build_model_custom(trial,in_features)
    # reset weights between each hyperparameter search
    Model.apply(reset_weights)
    # Train and validate model with given hyperparameters
    train_for_hy = train_and_validate_FFNN(Params,Model,Data_train,Data_Val,Scaler_Y,epoch,save_model,save_path)
    
    return train_for_hy["best_loss"]


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Functions related to ANN DS13,DS123,DS23, DS3, DS4,DS124,DS24:

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class model_DS123(nn.Module):
    '''
    Despite the name, this function is used for building up all ANN models with numeric inputs and 1 CNN input.
    Thus, DS13, DS123, DS23,DS3,DS4,DS124,DS24. 
    
    '''
    def __init__(self,trial, input_dim_desc, X_length,stride_CNN,
                 conv_dilation1,padding1, max_pool_kernel_size):
        super(model_DS123,self).__init__()
        
        # Hyperparameters:
        self.conv1_filters = trial.suggest_int("conv1_filters", 2, 32)
        self.conv2_filters = trial.suggest_int("conv2_filters", 2, 32)
        Kernel_size1 = trial.suggest_int("Kernel_size1", 2, 32)
        
        dropout_FFNN = trial.suggest_uniform("dropout_FFNN", 0, 0.6)
        dropout_CNN = trial.suggest_uniform("dropout_CNN", 0, 0.6)
        
        #set_trace()
        self.fc_neurons_out = trial.suggest_int("FC_after_CNN", 50, 300)
        self.desc_regressor_out1 = trial.suggest_int("FC_After_DS12", 1, 30)
        self.latent_variable1 = trial.suggest_int("FC_Concatenation", 50, 200)
        
        self.max_pool_kernel_size = max_pool_kernel_size
        
        ## Calculate output size:
        # After filter
        self.OUT_SIZE_1 = int(( (X_length + 2*padding1 - conv_dilation1*(Kernel_size1-1)-1) / stride_CNN) + 1    )
        # After Maxpool
        self.OUT_SIZE_1 = int(( (self.OUT_SIZE_1 + 2*padding1 - conv_dilation1*(self.max_pool_kernel_size-1)-1) / self.max_pool_kernel_size) + 1    )
        
        self.OUT_SIZE_2 = int((  (self.OUT_SIZE_1 + 2*padding1 - conv_dilation1*(self.max_pool_kernel_size-1)-1) / stride_CNN) + 1    )
        self.OUT_SIZE_2 = int((  (self.OUT_SIZE_2 + 2*padding1 - conv_dilation1*(Kernel_size1-1)-1) / self.max_pool_kernel_size) + 1    )
        
        self.maxpool=nn.MaxPool1d(self.max_pool_kernel_size, self.max_pool_kernel_size)
        ######### Define network
        self.CNN1 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters,
                                            kernel_size=Kernel_size1,stride=stride_CNN,dilation = conv_dilation1, padding = padding1),
                                      nn.BatchNorm1d(self.conv1_filters),
                                      nn.ReLU(),
                                      self.maxpool,
                                      nn.Dropout(p = dropout_CNN))
        
        self.CNN2 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters,out_channels = self.conv2_filters,
                                            kernel_size=Kernel_size1,stride=stride_CNN,dilation = conv_dilation1, padding = padding1),
                                      nn.BatchNorm1d(self.conv2_filters),
                                      nn.ReLU(),
                                      self.maxpool,
                                      nn.Dropout(p = dropout_CNN))
        
        self.regressor = nn.Sequential(nn.Linear(self.conv2_filters*self.OUT_SIZE_2 ,  self.fc_neurons_out) ,
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        self.desc_regressor1 = nn.Sequential(nn.Linear(in_features=input_dim_desc, out_features= self.desc_regressor_out1),
                                            nn.BatchNorm1d(self.desc_regressor_out1),
                                            nn.ReLU(),
                                            nn.Dropout(p = dropout_FFNN))
        
        self.ensemble_regressor = nn.Sequential(nn.Linear(self.desc_regressor_out1 +  self.fc_neurons_out , self.latent_variable1),
                                                nn.BatchNorm1d(self.latent_variable1),
                                                nn.ReLU(),
                                                nn.Dropout(p = dropout_FFNN),
                                                nn.Linear(self.latent_variable1 , 3))
   
    def forward(self,X_desc, X_seq):
        features_total = []
        features = self.CNN1(X_seq)
        features = self.CNN2(features)
        #print(features.shape)
        #print(self.OUT_SIZE_2)
        
        fc_in = features.view(-1, self.conv2_filters*self.OUT_SIZE_2)
        out_seq = self.regressor(fc_in)
        features_total.append(out_seq)
        x_des = self.desc_regressor1(X_desc)
        features_total.append(x_des)
        
        features_final = torch.cat(features_total,axis=1)
        out = self.ensemble_regressor(features_final)
        return  out

    

class model_DS123_build(nn.Module):
    def __init__(self,params, input_dim_desc, X_length,stride_CNN,
                 conv_dilation1,padding1, max_pool_kernel_size):
        super(model_DS123_build,self).__init__()
        
        # Hyperparameters:
        self.conv1_filters = params["conv1_filters"]
        self.conv2_filters = params["conv2_filters"]
        Kernel_size1 = params["Kernel_size1"]
        
        dropout_FFNN = params["dropout_FFNN"]
        dropout_CNN = params["dropout_CNN"]
        
        #set_trace()
        self.fc_neurons_out = params["FC_after_CNN"]
        self.desc_regressor_out1 = params["FC_After_DS12"]
        self.latent_variable1 = params["FC_Concatenation"]
        
        self.max_pool_kernel_size = max_pool_kernel_size
        
        ## Calculate output size for conv layer 1:
        # After filter
        self.OUT_SIZE_1 = int(( (X_length + 2*padding1 - conv_dilation1*(Kernel_size1-1)-1) / stride_CNN) + 1    )
        # After Maxpool
        self.OUT_SIZE_1 = int(( (self.OUT_SIZE_1 + 2*padding1 - conv_dilation1*(self.max_pool_kernel_size-1)-1) / self.max_pool_kernel_size) + 1    )
        ## Calculate output size for conv layer 2:
        # Filter:
        self.OUT_SIZE_2 = int((  (self.OUT_SIZE_1 + 2*padding1 - conv_dilation1*(self.max_pool_kernel_size-1)-1) / stride_CNN) + 1    )
        #Maxpool:
        self.OUT_SIZE_2 = int((  (self.OUT_SIZE_2 + 2*padding1 - conv_dilation1*(Kernel_size1-1)-1) / self.max_pool_kernel_size) + 1    )
        
        self.maxpool=nn.MaxPool1d(self.max_pool_kernel_size, self.max_pool_kernel_size)
        ######### Define network
        self.CNN1 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters,
                                            kernel_size=Kernel_size1,stride=stride_CNN,dilation = conv_dilation1, padding = padding1),
                                      nn.BatchNorm1d(self.conv1_filters),
                                      nn.ReLU(),
                                      self.maxpool,
                                      nn.Dropout(p = dropout_CNN))
        
        self.CNN2 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters,out_channels = self.conv2_filters,
                                            kernel_size=Kernel_size1,stride=stride_CNN,dilation = conv_dilation1, padding = padding1),
                                      nn.BatchNorm1d(self.conv2_filters),
                                      nn.ReLU(),
                                      self.maxpool,
                                      nn.Dropout(p = dropout_CNN))
        
        self.regressor = nn.Sequential(nn.Linear(self.conv2_filters*self.OUT_SIZE_2 ,  self.fc_neurons_out) ,
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        self.desc_regressor1 = nn.Sequential(nn.Linear(in_features=input_dim_desc, out_features= self.desc_regressor_out1),
                                            nn.BatchNorm1d(self.desc_regressor_out1),
                                            nn.ReLU(),
                                            nn.Dropout(p = dropout_FFNN))
        
        self.ensemble_regressor = nn.Sequential(nn.Linear(self.desc_regressor_out1 +  self.fc_neurons_out , self.latent_variable1),
                                                nn.BatchNorm1d(self.latent_variable1),
                                                nn.ReLU(),
                                                nn.Dropout(p = dropout_FFNN),
                                                nn.Linear(self.latent_variable1 , 3))
   
    def forward(self,X_desc, X_seq):
        features_total = []
        features = self.CNN1(X_seq)
        features = self.CNN2(features)
        #print(features.shape)
        #print(self.OUT_SIZE_2)
        
        fc_in = features.view(-1, self.conv2_filters*self.OUT_SIZE_2)
        out_seq = self.regressor(fc_in)
        features_total.append(out_seq)
        x_des = self.desc_regressor1(X_desc)
        features_total.append(x_des)
        
        features_final = torch.cat(features_total,axis=1)
        out = self.ensemble_regressor(features_final)
        return  out


def test_1CNN(network,params, test_loader,scaler_Y,save_path): 
    network.load_state_dict(torch.load(save_path))
    network.train(False)
    test_loss = 0
    tmp_assay1 = 0; tmp_assay2 = 0; tmp_assay3 = 0; tmp_assay4 = 0   
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # Get inputs
            inputs1,inputs2, targets = data
            # Perform forward pass
            network.double()
            outputs_scaled = network(inputs1,inputs2)
            # Compute loss
            loss = loss_fc(outputs_scaled, targets)    
            output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
            target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
            test_loss += np.sqrt(loss_fc(output_tensor,target_tensor).item()) 
        
        #test_loss /= test_loader.__len__()
        tmp_assay1 += np.sqrt(loss_fc(output_tensor[:,0],target_tensor[:,0]).item())
        tmp_assay2 += np.sqrt(loss_fc(output_tensor[:,1],target_tensor[:,1]).item())
        tmp_assay3 += np.sqrt(loss_fc(output_tensor[:,2],target_tensor[:,2]).item())
        
    return {"best_rmse":test_loss,"tmp_assay1": tmp_assay1, "tmp_assay2": tmp_assay2, "tmp_assay3": tmp_assay3, "output_tensor": output_tensor, "True_Y":target_tensor}            


def train_and_validate_1CNN(Params,Model,Data_train,Data_Val,scaler_Y,EPOCH,save_model,save_path):
    # Initiate dataloaders with batch_size as hyperparameter:
    train_loader = DataLoader(dataset = Data_train,batch_size=Params["Batch_Size"],shuffle=True,num_workers=0,drop_last = True)
    val_loader = DataLoader(dataset = Data_Val,batch_size=Params["Batch_Size"],shuffle=True,drop_last = True)
 
    train_loss_save = []
    val_loss_save = []
    Model.train(True)
    best_loss = 10000
    # Initialize optimizer
    optimizer = torch.optim.Adam(Model.parameters(), lr=Params['lr'],weight_decay= Params['wd'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(0, EPOCH):
        # Set current loss value
        train_loss = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs1,inputs2, targets = data            
            
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            Model.double()
            outputs_scaled = Model(inputs1,inputs2)
            # Compute loss
            loss = loss_fc(outputs_scaled, targets)    
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # conversion back before reporting loss
            output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
            target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
            train_loss += loss_fc(output_tensor,target_tensor).item()
            
        #Done with training. Now test(validate):    
        Model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # Get inputs
                inputs1,inputs2, targets = data 
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                Model.double()
                outputs_scaled = Model(inputs1,inputs2)
                # Compute loss
                loss = loss_fc(outputs_scaled, targets)    
                #Scaled back-loss
                output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
                target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
                test_loss += loss_fc(output_tensor,target_tensor).item()
                
        # Print
        train_loss /= train_loader.__len__()
        test_loss /= val_loader.__len__()
        #scheduler.step(test_loss)
        train_loss = np.sqrt(train_loss)
        test_loss = np.sqrt(test_loss)
        
        train_loss_save.append(train_loss)
        val_loss_save.append(test_loss)
        
        #if EPOCH % 1 == 0:
        #   print('Epoch %i: Train RMSE: %.3f Val RMSE: %.3f' % (epoch, train_loss, test_loss))
            
        if test_loss < best_loss:
            if save_model:
                torch.save(Model.state_dict(), save_path)
            best_loss = test_loss
            
    return {"best_loss": best_loss, "train_loss": train_loss_save, "val_loss": val_loss_save}

def objective_DS123(trial,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path,X_length,max_pool_kernel_size):
    '''
    Optuna wrapper function for hyperparameter optimization
    
    '''
    
    Params = {"lr":trial.suggest_loguniform("lr",1e-6,1e-2),
              "Batch_Size":trial.suggest_int("Batch_Size", 4, 64),
              "wd":trial.suggest_uniform("wd",0,1e-1)
             }
    dimension = next(iter(DataLoader(Data_train)))[0].shape[1]
    
    Model = model_DS123(trial,input_dim_desc = dimension, X_length = X_length,stride_CNN = 1,
                 conv_dilation1 = 1,padding1 = 0, max_pool_kernel_size = max_pool_kernel_size)
    
    train_for_hy = train_and_validate_1CNN(Params,Model,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path)
    
    return train_for_hy["best_loss"]


class Dataset_seq_embeddings(Dataset):
    '''
    Pytorch dataset class that handles 1 numeric input, 1 sequential input and target
    
    '''
    def __init__(self, X_descriptors, X_sequence,Y):
        self.X_descriptors = X_descriptors
        self.Y = Y
        self.X_sequence = X_sequence
        
    def __getitem__(self, index):
        # return the seq and label 
        seq = torch.tensor(self.X_sequence.iloc[index])
        RA = torch.tensor(self.Y.iloc[index])
        desc = torch.tensor(pd.DataFrame(self.X_descriptors.iloc[index]).T.values)
        return torch.squeeze(desc),torch.unsqueeze(seq,0), RA

    def __len__(self):
        return(len(self.X_descriptors))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Functions related to ANN DS34,DS134,DS1234,DS234:

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Dataset_all_conc(Dataset):
    def __init__(self,X_desc, X_sequence,X_smiles,Y):
        self.Y = Y
        self.X_sequence = X_sequence
        self.X_sequence_smiles = X_smiles
        self.X_desc = X_desc
        
        
        
        
    def __getitem__(self, index):
        # return the seq and label 
        seq = torch.tensor(self.X_sequence.iloc[index])
        seq_smiles = torch.tensor(self.X_sequence_smiles.iloc[index])
        X_desc =  torch.tensor(self.X_desc.iloc[index])
        RA = torch.tensor(self.Y.iloc[index])
        
        total_features = torch.cat([X_desc,seq,seq_smiles])
        return total_features.float(), RA

    def __len__(self):
        return(len(self.Y))


def objective_DS1234(trial,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path,input_dim_desc):
    Params = {"lr":trial.suggest_loguniform("lr",1e-6,1e-2),
              "Batch_Size":trial.suggest_int("Batch_Size", 12, 64),
              "wd":trial.suggest_uniform("wd",0,1e-1)
             }

    
    Model = model_DS1234(trial,input_dim_desc = input_dim_desc, X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 6)
    Model.apply(reset_weights)
    train_for_hy = train_and_validate(Params,Model,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path,trial)
    return train_for_hy["best_loss"]


def objective_DS1234_fixed_batch_size(trial,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path,input_dim_desc):
    Params = {"lr":trial.suggest_loguniform("lr",1e-6,1e-2),
              "Batch_Size": 2,
              "wd":trial.suggest_uniform("wd",0,1e-1)
             }

    
    Model = model_DS1234(trial,input_dim_desc = input_dim_desc, X_length_DS3 = 1280,X_length_DS4 = 100,stride_CNN = 1,
                 conv_dilation1 = 1,padding_DS3 = 0,padding_DS4 = 0, max_pool_kernel_size3 = 6)
    Model.apply(reset_weights)
    train_for_hy = train_and_validate(Params,Model,Data_train,Data_Val,Scaler_Y,EPOCH,save_model,save_path,trial)
    return train_for_hy["best_loss"]


def train_and_validate(Params,Model,Data_train,Data_Val,scaler_Y,EPOCH,save_model,save_path, trial):
    # Initiate dataloaders with batch_size as hyperparameter:
    train_loader = DataLoader(dataset = Data_train,batch_size=Params["Batch_Size"],shuffle=True,num_workers=0,drop_last = True)
    val_loader = DataLoader(dataset = Data_Val,batch_size=Params["Batch_Size"],shuffle=True,drop_last = True)
 
    train_loss_save = []
    val_loss_save = []
    Model.train(True)
    best_loss = 10000
    # Initialize optimizer
    optimizer = torch.optim.Adam(Model.parameters(), lr=Params['lr'],weight_decay= Params['wd'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(0, EPOCH):
        # Set current loss value
        train_loss = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs,targets = data            
            
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            Model.float()
            outputs_scaled = Model(inputs)
            # Compute loss
            loss = loss_fc(outputs_scaled, targets)    
            # Perform backward pass            
            loss.backward()
            # Perform optimization
            optimizer.step()
            # conversion back before reporting loss
            output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
            target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
            train_loss += loss_fc(output_tensor,target_tensor).item()
            
        #Done with training. Now test(validate):    
        Model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # Get inputs
                inputs,targets = data 
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                Model.float()
                outputs_scaled = Model(inputs)
                #Scaled back-loss
                output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
                target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
                test_loss += loss_fc(output_tensor,target_tensor).item()
                
        # Print
        train_loss /= train_loader.__len__()
        test_loss /= val_loader.__len__()
        #scheduler.step(test_loss)
        train_loss = np.sqrt(train_loss)
        test_loss = np.sqrt(test_loss)
        
        train_loss_save.append(train_loss)
        val_loss_save.append(test_loss)
        
        #if EPOCH % 100 == 0:
        #    print('Epoch %i: Train RMSE: %.3f Val RMSE: %.3f' % (epoch, train_loss, test_loss))
            
        if test_loss < best_loss:
            if save_model:
                torch.save(Model.state_dict(), save_path)
            best_loss = test_loss
            
        # Add prune mechanism
        if trial != "None":
            trial.report(test_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    return {"best_loss": best_loss, "train_loss": train_loss_save, "val_loss": val_loss_save}







def test_FFNN(network,params, test_loader,scaler_Y,save_path,Y_data_for_index): 
    network.load_state_dict(torch.load(save_path))
    network.train(False)
    test_loss = 0
    tmp_assay1 = 0; tmp_assay2 = 0; tmp_assay3 = 0; tmp_assay4 = 0   
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # Get inputs
            inputs, targets = data
            # Perform forward pass
            #network.float()
            outputs_scaled = network(inputs)
            # Compute loss
            loss = loss_fc(outputs_scaled, targets)    
            output_tensor = torch.Tensor(scaler_Y.inverse_transform(outputs_scaled.detach().numpy()))
            target_tensor = torch.Tensor(scaler_Y.inverse_transform(targets.detach().numpy()))
            test_loss += np.sqrt(loss_fc(output_tensor,target_tensor).item()) 
        
        #test_loss /= test_loader.__len__()
        tmp_assay1 += np.sqrt(loss_fc(output_tensor[:,0],target_tensor[:,0]).item())
        tmp_assay2 += np.sqrt(loss_fc(output_tensor[:,1],target_tensor[:,1]).item())
        tmp_assay3 += np.sqrt(loss_fc(output_tensor[:,2],target_tensor[:,2]).item())
    
    output_pd = pd.DataFrame(output_tensor,columns = Y_data_for_index.columns).set_index(Y_data_for_index.index)
    target_pd = pd.DataFrame(target_tensor,columns = Y_data_for_index.columns).set_index(Y_data_for_index.index)
    
    return {"best_rmse":test_loss,"tmp_assay1": tmp_assay1, "tmp_assay2": tmp_assay2, "tmp_assay3": tmp_assay3, "output_tensor": output_pd, "True_Y":target_pd}   




class model_DS1234_build(nn.Module):
    def __init__(self,params, input_dim_desc, X_length_DS3,X_length_DS4,stride_CNN,
                 conv_dilation1,padding_DS3,padding_DS4, max_pool_kernel_size3):
        super(model_DS1234_build,self).__init__()
        
        
        self.input_dim_desc = input_dim_desc
        # Hyperparameters:
        self.conv1_filters_DS3 = params["conv1_filters_DS3"]
        self.conv2_filters_DS3 = params["conv2_filters_DS3"]

        self.conv1_filters_DS4 = params["conv1_filters_DS4"]
        self.conv2_filters_DS4 = params["conv2_filters_DS4"]
        
        
        
        Kernel_size1_DS3 = params["Kernel_size1_DS3"]
        Kernel_size1_DS4 = params["Kernel_size1_DS4"]
        
        
        dropout_FFNN = params["dropout_FFNN_DS3"]
        dropout_CNN_DS3 = params["dropout_CNN_DS3"]
        dropout_CNN_DS4 = params["dropout_CNN_DS4"]
        
        self.fc_neurons_out_DS3 = params["FC_after_CNN_DS3"]
        self.fc_neurons_out_DS4 = params["FC_after_CNN_DS4"]
        
        self.desc_regressor_DS12 = params["FC_After_DS12"]
        self.latent_variable1 = params["FC_Concatenation"]
        
        
        ## Input defined maxpool kernel size
        self.max_pool_kernel_size3 = max_pool_kernel_size3
        ## Default maxpool kernel size is to use filter kernel size
        #self.max_pool_kernel_size3 = Kernel_size1_DS3
        #self.max_pool_kernel_size4 = Kernel_size1_DS4
        
        
        ## Calculate output size DS3:
        # After filter
        self.OUT_SIZE_1_DS3 = int(( (X_length_DS3 + 2*padding_DS3 - conv_dilation1*(Kernel_size1_DS3-1)-1) / stride_CNN) + 1    )
        # After Maxpool
        self.OUT_SIZE_1_DS3 = int(( (self.OUT_SIZE_1_DS3 + 2*padding_DS3 - conv_dilation1*(self.max_pool_kernel_size3-1)-1) / self.max_pool_kernel_size3) + 1    )
        
        self.OUT_SIZE_2_DS3 = int((  (self.OUT_SIZE_1_DS3 + 2*padding_DS3 - conv_dilation1*(Kernel_size1_DS3-1)-1) / stride_CNN) + 1    )
        self.OUT_SIZE_2_DS3 = int((  (self.OUT_SIZE_2_DS3 + 2*padding_DS3 - conv_dilation1*(self.max_pool_kernel_size3-1)-1) / self.max_pool_kernel_size3) + 1    )
        #print(self.OUT_SIZE_2_DS3)
        
        ## Calculate output size DS4:
        # After filter
        self.OUT_SIZE_1_DS4 = int(( (X_length_DS4 + 2*padding_DS4 - conv_dilation1*(Kernel_size1_DS4-1)-1) / stride_CNN) + 1    )
        self.OUT_SIZE_2_DS4 = int((  (self.OUT_SIZE_1_DS4 + 2*padding_DS4 - conv_dilation1*(Kernel_size1_DS4-1)-1) / stride_CNN) + 1    )
        
        
        
        # Note that in order to calculate SHAP values we have to define maxpool / Activation everytime and NOT re-use a predefined maxpool/activation. Bug in SHAP. 
        ######### Define network
        ######## CNN Layers
        self.CNN1_DS3 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters_DS3,
                                            kernel_size=Kernel_size1_DS3,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS3),
                                      nn.BatchNorm1d(self.conv1_filters_DS3),
                                      nn.ReLU(),
                                      nn.MaxPool1d(self.max_pool_kernel_size3),
                                      nn.Dropout(p = dropout_CNN_DS3))
        
        self.CNN2_DS3 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters_DS3,out_channels = self.conv2_filters_DS3,
                                            kernel_size=Kernel_size1_DS3,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS3),
                                      nn.BatchNorm1d(self.conv2_filters_DS3),
                                      nn.ReLU(),
                                      nn.MaxPool1d(self.max_pool_kernel_size3),
                                      nn.Dropout(p = dropout_CNN_DS3))
        
        
        self.CNN1_DS4 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters_DS4,
                                            kernel_size=Kernel_size1_DS4,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS4),
                                      nn.BatchNorm1d(self.conv1_filters_DS4),
                                      nn.ReLU(),
                                      nn.Dropout(p = dropout_CNN_DS4))
        
        self.CNN2_DS4 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters_DS4,out_channels = self.conv2_filters_DS4,
                                            kernel_size=Kernel_size1_DS4,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS4),
                                      nn.BatchNorm1d(self.conv2_filters_DS4),
                                      nn.ReLU(),
                                      nn.Dropout(p = dropout_CNN_DS4))
        
        ################## FC Layers
        
        self.Regressor_DS3 = nn.Sequential(nn.Linear(self.conv2_filters_DS3*self.OUT_SIZE_2_DS3, self.fc_neurons_out_DS3) ,
                                      nn.BatchNorm1d(self.fc_neurons_out_DS3),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        self.Regressor_DS4 = nn.Sequential(nn.Linear(self.conv2_filters_DS4*self.OUT_SIZE_2_DS4, self.fc_neurons_out_DS4) ,
                                      nn.BatchNorm1d(self.fc_neurons_out_DS4),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        
        self.Regressor_DS12 = nn.Sequential(nn.Linear(self.input_dim_desc , self.desc_regressor_DS12 ) ,
                                       nn.BatchNorm1d(self.desc_regressor_DS12 ),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        
        self.concatenated_regressor = nn.Sequential(nn.Linear(self.fc_neurons_out_DS4+ self.fc_neurons_out_DS3 + self.desc_regressor_DS12,self.latent_variable1),
                                                nn.BatchNorm1d(self.latent_variable1),
                                                nn.ReLU(),
                                                nn.Dropout(p = dropout_FFNN),
                                                nn.Linear(self.latent_variable1 , 3))
   
    def forward(self,x):
        
        x_desc = x[:,:self.input_dim_desc]
        
        x_DS3 = x[:,self.input_dim_desc: self.input_dim_desc+1280]
        x_DS3 = torch.unsqueeze(x_DS3,1)
        x_DS4 = x[:,self.input_dim_desc+1280:]
        x_DS4 = torch.unsqueeze(x_DS4,1)
        
        features_total = []
        ## DS12 features:
        desc_features = self.Regressor_DS12(x_desc)
        # DS3 features
        
        features_DS3 = self.CNN1_DS3(x_DS3)
        features_DS3 = self.CNN2_DS3(features_DS3)   
        fc_in_DS3 = features_DS3.view(-1, self.OUT_SIZE_2_DS3*self.conv2_filters_DS3)
        out_DS3 = self.Regressor_DS3(fc_in_DS3)
        
     
        # DS4 features
        features_DS4 = self.CNN1_DS4(x_DS4)
        features_DS4 = self.CNN2_DS4(features_DS4)   
        #print(features_DS4.shape)
        #print(self.OUT_SIZE_2_DS4)
        #print(self.OUT_SIZE_2_DS4*self.conv2_filters_DS4)
        
        
        fc_in_DS4 = features_DS4.view(-1,self.OUT_SIZE_2_DS4*self.conv2_filters_DS4)
        out_DS4 = self.Regressor_DS4(fc_in_DS4)
        
        ## 
        
        features_total.append(desc_features)
        features_total.append(out_DS3)
        features_total.append(out_DS4)
        
        features_final = torch.cat(features_total,axis=1)
        out =  self.concatenated_regressor(features_final)
        return out


class model_DS1234(nn.Module):
    def __init__(self,trial, input_dim_desc, X_length_DS3,X_length_DS4,stride_CNN,
                 conv_dilation1,padding_DS3,padding_DS4, max_pool_kernel_size3):
        super(model_DS1234,self).__init__()
        self.input_dim_desc = input_dim_desc
        # Hyperparameters:
        self.conv1_filters_DS3 = trial.suggest_int("conv1_filters_DS3", 2, 32)
        self.conv2_filters_DS3 = trial.suggest_int("conv2_filters_DS3", 2, 32)
        
        
        self.conv1_filters_DS4 = trial.suggest_int("conv1_filters_DS4", 2, 16)
        self.conv2_filters_DS4 = trial.suggest_int("conv2_filters_DS4", 2, 16)
        
        
        Kernel_size1_DS3 = trial.suggest_int("Kernel_size1_DS3", 2, 32)
        Kernel_size1_DS4 = trial.suggest_int("Kernel_size1_DS4", 2, 16)
        
        #Kernel_size1_DS3 = 5
        #Kernel_size1_DS4 = 5
        
        ## Set manually:
        self.max_pool_kernel_size3 = max_pool_kernel_size3
        
        # Let default rule:
        #self.max_pool_kernel_size3 = Kernel_size1_DS3
        #self.max_pool_kernel_size4 = Kernel_size1_DS4
        
        self.maxpool_DS3 = nn.MaxPool1d(self.max_pool_kernel_size3)
       
        
        
        
        
        dropout_FFNN = trial.suggest_uniform("dropout_FFNN_DS3", 0, 0.6)
        dropout_CNN_DS3 = trial.suggest_uniform("dropout_CNN_DS3", 0, 0.6)
        dropout_CNN_DS4 = trial.suggest_uniform("dropout_CNN_DS4", 0, 0.6)
        
        self.fc_neurons_out_DS3 = trial.suggest_int("FC_after_CNN_DS3", 50, 300)
        self.fc_neurons_out_DS4 = trial.suggest_int("FC_after_CNN_DS4", 50, 300)
        
        self.desc_regressor_DS12 = trial.suggest_int("FC_After_DS12", 1, 30)
        self.latent_variable1 = trial.suggest_int("FC_Concatenation", 20, 100)
        
       
        
        ## Calculate output size DS3:
        # After filter
        self.OUT_SIZE_1_DS3 = int(( (X_length_DS3 + 2*padding_DS3 - conv_dilation1*(Kernel_size1_DS3-1)-1) / stride_CNN) + 1    )
        # After Maxpool
        self.OUT_SIZE_1_DS3 = int(( (self.OUT_SIZE_1_DS3 + 2*padding_DS3 - conv_dilation1*(self.max_pool_kernel_size3-1)-1) / self.max_pool_kernel_size3) + 1    )
        
        self.OUT_SIZE_2_DS3 = int((  (self.OUT_SIZE_1_DS3 + 2*padding_DS3 - conv_dilation1*(Kernel_size1_DS3-1)-1) / stride_CNN) + 1    )
        self.OUT_SIZE_2_DS3 = int((  (self.OUT_SIZE_2_DS3 + 2*padding_DS3 - conv_dilation1*(self.max_pool_kernel_size3-1)-1) / self.max_pool_kernel_size3) + 1    )
        #print(self.OUT_SIZE_2_DS3)
        
        ## Calculate output size DS4:
        # After filter
        self.OUT_SIZE_1_DS4 = int(( (X_length_DS4 + 2*padding_DS4 - conv_dilation1*(Kernel_size1_DS4-1)-1) / stride_CNN) + 1    )
        # After Maxpool

        self.OUT_SIZE_2_DS4 = int((  (self.OUT_SIZE_1_DS4 + 2*padding_DS4 - conv_dilation1*(Kernel_size1_DS4-1)-1) / stride_CNN) + 1    )
        #print(self.OUT_SIZE_2_DS4)
        
        
        
      
        
        ######### Define network
        ######## CNN Layers
        self.CNN1_DS3 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters_DS3,
                                            kernel_size=Kernel_size1_DS3,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS3),
                                      nn.BatchNorm1d(self.conv1_filters_DS3),
                                      nn.ReLU(),
                                      self.maxpool_DS3,
                                      #nn.MaxPool1d(3),
                                      nn.Dropout(p = dropout_CNN_DS3))
        
        self.CNN2_DS3 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters_DS3,out_channels = self.conv2_filters_DS3,
                                            kernel_size=Kernel_size1_DS3,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS3),
                                      nn.BatchNorm1d(self.conv2_filters_DS3),
                                      nn.ReLU(),
                                      self.maxpool_DS3,
                                      #nn.MaxPool1d(3),
                                      nn.Dropout(p = dropout_CNN_DS3))
        
        
        self.CNN1_DS4 = nn.Sequential(nn.Conv1d(in_channels = 1,out_channels = self.conv1_filters_DS4,
                                            kernel_size=Kernel_size1_DS4,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS4),
                                      nn.BatchNorm1d(self.conv1_filters_DS4),
                                      nn.ReLU(),
                                      #nn.MaxPool1d(3),
                                      nn.Dropout(p = dropout_CNN_DS4))
        
        self.CNN2_DS4 = nn.Sequential(nn.Conv1d(in_channels = self.conv1_filters_DS4,out_channels = self.conv2_filters_DS4,
                                            kernel_size=Kernel_size1_DS4,stride=stride_CNN,dilation = conv_dilation1, padding = padding_DS4),
                                      nn.BatchNorm1d(self.conv2_filters_DS4),
                                      nn.ReLU(),
                                      #nn.MaxPool1d(3),
                                      nn.Dropout(p = dropout_CNN_DS4))
        
        ################## FC Layers
        
        self.Regressor_DS3 = nn.Sequential(nn.Linear(self.conv2_filters_DS3*self.OUT_SIZE_2_DS3, self.fc_neurons_out_DS3) ,
                                      nn.BatchNorm1d(self.fc_neurons_out_DS3),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        self.Regressor_DS4 = nn.Sequential(nn.Linear(self.conv2_filters_DS4*self.OUT_SIZE_2_DS4, self.fc_neurons_out_DS4) ,
                                      nn.BatchNorm1d(self.fc_neurons_out_DS4),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        
        self.Regressor_DS12 = nn.Sequential(nn.Linear(self.input_dim_desc, self.desc_regressor_DS12 ) ,
                                       nn.BatchNorm1d(self.desc_regressor_DS12 ),
                                       nn.ReLU(), 
                                       nn.Dropout(p = dropout_FFNN)
                                      )
        
        self.concatenated_regressor = nn.Sequential(nn.Linear(self.fc_neurons_out_DS4+ self.fc_neurons_out_DS3 + self.desc_regressor_DS12,self.latent_variable1),
                                                nn.BatchNorm1d(self.latent_variable1),
                                                nn.ReLU(),
                                                nn.Dropout(p = dropout_FFNN),
                                                nn.Linear(self.latent_variable1 , 3))
   
    def forward(self,x):
        x_desc = x[:,:self.input_dim_desc]
        x_DS3 = x[:,self.input_dim_desc: self.input_dim_desc+1280]
        x_DS3 = torch.unsqueeze(x_DS3,1)
        
        x_DS4 = x[:,self.input_dim_desc+1280:]
        x_DS4 = torch.unsqueeze(x_DS4,1)
        features_total = []
        ## DS12 features:
        desc_features = self.Regressor_DS12(x_desc)
        # DS3 features
        
        features_DS3 = self.CNN1_DS3(x_DS3)
        features_DS3 = self.CNN2_DS3(features_DS3)   
        #print(features_DS3.shape)
        fc_in_DS3 = features_DS3.view(-1, self.OUT_SIZE_2_DS3*self.conv2_filters_DS3)
        out_DS3 = self.Regressor_DS3(fc_in_DS3)
        
     
        # DS4 features
        features_DS4 = self.CNN1_DS4(x_DS4)
        features_DS4 = self.CNN2_DS4(features_DS4)   
        #print(features_DS4.shape)
        #print(self.OUT_SIZE_2_DS4)
        #print(self.OUT_SIZE_2_DS4*self.conv2_filters_DS4)
        
        
        fc_in_DS4 = features_DS4.view(-1,self.OUT_SIZE_2_DS4*self.conv2_filters_DS4)
        out_DS4 = self.Regressor_DS4(fc_in_DS4)
        
        ## 
        
        features_total.append(desc_features)
        features_total.append(out_DS3)
        features_total.append(out_DS4)
        
        features_final = torch.cat(features_total,axis=1)
        out =  self.concatenated_regressor(features_final)
        return out

    

