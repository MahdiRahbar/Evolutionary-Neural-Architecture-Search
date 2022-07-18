# Author: Mahdi
# Created: 1:04 PM, July 25th 2020
# Github: Github.com/MahdiRahbar

import pandas as pd
import numpy as np 
from bitstring import BitArray
import ast
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




def data_import(data_path):
    data = pd.read_csv(data_path)
    return data

def Bin_to_Dec(input_chrom, split_points):
    '''
    split point is considered as a list of inputs which can also kinda show the number of gens! 

    4 bits for framing --- 5 bits for each units  OVERALL=====> 14 gens 

    '''
    input_chrom = ast.literal_eval(input_chrom)
    # input_chrom = input_chrom[int(i) for i in input_chrom]
    window_size_bits = BitArray(input_chrom[:split_points[0]])
    num_units_1_bits = BitArray(input_chrom[split_points[0]:split_points[1]])
    num_units_2_bits = BitArray(input_chrom[split_points[1]:split_points[2]])
    num_dense1_bits = BitArray(input_chrom[split_points[2]:split_points[3]])
    num_dense2_bits = BitArray(input_chrom[split_points[3]:])

    window_size = window_size_bits.uint
    num_units_1 = num_units_1_bits.uint
    num_units_2 = num_units_2_bits.uint
    num_dense1 = num_dense1_bits.uint
    num_dense2 = num_dense2_bits.uint

    # The below section has to be changed if the framing size or LSTM units were changed significantly!
    if window_size < 1:
        print("we are in first!")
        window_size = np.random.randint(13) + 2
    if num_units_1 < 1:
        print("we are in second")
        num_units_1 = np.random.randint(29) + 2
    if num_units_2 < 1:
        num_units_2 = np.random.randint(29) + 2
        print("Third")
    if num_dense1 < 1:
        num_dense1 = np.random.randint(61) + 2
        print("we are in Fouth")
    if num_dense2 < 1:
        num_dense2 = np.random.randint(61) + 2

    return window_size, num_units_1, num_units_2, num_dense1, num_dense2

    

def seq_framing(seq_data , seq_len = 14, split_percentage = 20/100, output_normalizer = True , normalizer = True, shuffle = True):
    # X = []
    # Y = []
    # seq_data = np.array(seq_data)
    # for i in range(len(seq_data) - seq_len):
    #     x_data = []
        
    #     for j in range(seq_len):
    #         x_data.append(seq_data[i+j])      
            
    #     x_data = np.array(x_data)
    #     x_data = x_data.reshape(1,seq_len)
    #     Y.append(seq_data[i+seq_len])
        
    #     X.append(x_data)
        
    # X = np.array(X)
    # X = X.reshape((len(X),seq_len,1))
    # Y = np.array(Y)
    # Y = Y.reshape((len(Y),1))    
    # split_number = math.floor(0.2 * len(X))
    # X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
    # y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]
    
    if output_normalizer == True & normalizer == True  & shuffle == False: 
        X_scaler = MinMaxScaler()
        seq_data = np.array(seq_data).reshape(-1, 1)
        X_scaler.fit(seq_data)
        seq_data = X_scaler.transform(seq_data)

        X = []
        Y = []
        seq_data = np.array(seq_data)
        for i in range(len(seq_data) - seq_len):
            x_data = []
            
            for j in range(seq_len):
                x_data.append(seq_data[i+j])      
                
            x_data = np.array(x_data)
            x_data = x_data.reshape(1,seq_len)
            Y.append(seq_data[i+seq_len])
            
            X.append(x_data)
            
        X = np.array(X)
        X = X.reshape((len(X),seq_len,1))
        Y = np.array(Y)
        Y = Y.reshape((len(Y),1))    
        split_number = math.floor(0.2 * len(X))
        X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
        y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]
    
    elif output_normalizer == False & normalizer == True & shuffle == False:
        temp_seq_data = seq_data
        X_scaler = MinMaxScaler()
        seq_data = np.array(seq_data).reshape(-1, 1)
        X_scaler.fit(seq_data)
        seq_data = X_scaler.transform(seq_data)

        X = []
        Y = []
        temp_seq_data = np.array(temp_seq_data)
        seq_data = np.array(seq_data)
        for i in range(len(seq_data) - seq_len):
            x_data = []
            
            for j in range(seq_len):
                x_data.append(seq_data[i+j])      
                
            x_data = np.array(x_data)
            x_data = x_data.reshape(1,seq_len)
            Y.append(temp_seq_data[i+seq_len])
            
            X.append(x_data)
            
        X = np.array(X)
        X = X.reshape((len(X),seq_len,1))
        Y = np.array(Y)
        Y = Y.reshape((len(Y),1))    
        split_number = math.floor(0.2 * len(X))
        X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
        y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]

    elif output_normalizer == False & normalizer == False & shuffle == False:

        X = []
        Y = []
        seq_data = np.array(seq_data)
        for i in range(len(seq_data) - seq_len):
            x_data = []
            
            for j in range(seq_len):
                x_data.append(seq_data[i+j])      
                
            x_data = np.array(x_data)
            x_data = x_data.reshape(1,seq_len)
            Y.append(seq_data[i+seq_len])
            
            X.append(x_data)
            
        X = np.array(X)
        X = X.reshape((len(X),seq_len,1))
        Y = np.array(Y)
        Y = Y.reshape((len(Y),1))    
        split_number = math.floor(0.2 * len(X))
        X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
        y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]

        X_scaler = None
    
    elif output_normalizer == True & normalizer == True & shuffle == True:
        
        cases = seq_data['cases']
        Cumulative = seq_data['Cumulative_number']
        Day_Status = np.array(seq_data['Day_Status']).reshape(-1,1)
        Gathering_Potential = np.array(seq_data['Gathering_Potential']).reshape(-1,1)



        X_scaler = MinMaxScaler()
        data_array = np.array(cases).reshape(-1, 1)
        X_scaler.fit(data_array)
        X_Norm = X_scaler.transform(data_array)
        
        Cumulative_scaler = MinMaxScaler()
        data_array = np.array(Cumulative).reshape(-1, 1)
        Cumulative_scaler.fit(data_array)
        Cumulative_Norm = Cumulative_scaler.transform(data_array)

        seq_data = np.concatenate((X_Norm , Cumulative_Norm , Day_Status , Gathering_Potential), axis=1)
        

        X = []
        Y = []
        seq_data = np.array(seq_data)
        for i in range(seq_data.shape[0] - seq_len):
            x_data = []
            x_data = seq_data[i:i+seq_len]

            x_data = x_data.reshape(1,seq_len,seq_data.shape[1])
            Y.append(seq_data[i+seq_len,0])
            
            X.append(x_data)
            
        X = np.array(X)
        X = X.reshape((X.shape[0],seq_len,seq_data.shape[1]))
        Y = np.array(Y)
        Y = Y.reshape((len(Y),1))   


        #### The Type of splitting 
        split_number = math.floor(split_percentage * len(X))
        # X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
        # y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_percentage, random_state=42)
        # X_scaler = MinMaxScaler()
        # seq_data = np.array(seq_data).reshape(-1, 1)
        # X_scaler.fit(seq_data)
        # seq_data = X_scaler.transform(seq_data)

        # X = []
        # Y = []
        # seq_data = np.array(seq_data)
        # for i in range(len(seq_data) - seq_len):
        #     x_data = []
            
        #     for j in range(seq_len):
        #         x_data.append(seq_data[i+j])      
                
        #     x_data = np.array(x_data)
        #     x_data = x_data.reshape(1,seq_len)
        #     Y.append(seq_data[i+seq_len])
            
        #     X.append(x_data)
            
        # X = np.array(X)
        # X = X.reshape((len(X),seq_len,1))
        # Y = np.array(Y)
        # Y = Y.reshape((len(Y),1))    
        # split_number = math.floor(0.2 * len(X))
        # X_train, X_test = X[:len(X)-split_number], X[len(X)-split_number:]
        # y_train, y_test = Y[:len(Y)-split_number], Y[len(Y)-split_number:]

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_percentage, random_state=42)


    return X_train, X_test, y_train, y_test , X_scaler
        