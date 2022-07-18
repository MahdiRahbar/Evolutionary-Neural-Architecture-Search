# Author: Mahdi
# Created: 1:04 PM, July 25th 2020
# Github: Github.com/MahdiRahbar

import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Suppress messages from tensorflow
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.callbacks import EarlyStopping
sys.stderr = stderr

from data_preprocess import *


SEED = 2
 
tf.random.set_seed(
    SEED
)

def model(data, window_size, num_units_1, num_units_2 , num_dense1, num_dense2):

    if window_size < 1 or num_units_1 < 1 or num_units_2 < 1 or num_dense1 < 1 or num_dense2<1:
        return 100
    X_train, X_test, y_train, y_test, X_scaler = seq_framing(data, window_size, split_percentage = 20/100, output_normalizer = True , normalizer = True, shuffle= True)

    model = tf.keras.Sequential()
    model.add(keras.Input(shape=(X_train.shape[1],X_train.shape[2])))

    model.add(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units_1), return_sequences =True))
    model.add(keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units_2), return_sequences =False))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(units = num_dense1, activation = 'relu'))   # The number of units have to be picked in this section 
    model.add(keras.layers.Dense(units = num_dense2, activation = 'relu'))
    model.add(keras.layers.Dense(units = 1 , activation = 'relu'))

    # opt = SGD(lr=0.001)  # 'adam'
    opt = Adam(learning_rate=0.001)
    model.compile(loss = 'mean_squared_error', optimizer = opt, metrics=['mean_squared_error','mae'])  #  loss = 'mean_squared_error' | 'mean_squared_error'  | 'mae'  , metrics=['mean_squared_error','mae']
    # model.summary()
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose = 0, patience=50)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose = 0, save_best_only=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test) , epochs=200, verbose=0, callbacks=[mc])  #  [es,mc]    validation_split=0.2 , validation_data=(X_test, y_test) , callbacks=[es,mc]
    
    train_acc = model.evaluate(X_train, y_train, verbose=0)  # _, 
    test_acc = model.evaluate(X_test, y_test, verbose=0)  # _, 


    Best_Model_Info = pd.read_csv("Best_Model_Info.csv")
    if float(Best_Model_Info['test_acc']) > test_acc[0]:

        Best_Model_Info['test_acc'] = test_acc[0]
        Best_Model_Info['window_size'] = window_size
        Best_Model_Info['num_units_1'] = num_units_1
        Best_Model_Info['num_units_2'] = num_units_2
        Best_Model_Info['num_dense'] = num_dense1
        Best_Model_Info['num_dense'] = num_dense2

        Best_Model_Info.to_csv('Best_Model_Info.csv', index= False)

    # print(test_acc[0])
    return test_acc[0]  # test_mse





