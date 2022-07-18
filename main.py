# Author: Mahdi
# Created: 1:04 PM, July 25th 2020
# Github: Github.com/MahdiRahbar

import os
import logging 
import numpy as np
import sys
from data_preprocess import *
from model import * 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# def set_tf_loglevel(level):
#     if level >= logging.FATAL:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     if level >= logging.ERROR:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     if level >= logging.WARNING:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#     else:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#     logging.getLogger('tensorflow').setLevel(level)

# set_tf_loglevel(logging.FATAL)



DATA_PATH = '../data/Final_Iran_Data_27_8.csv'
SPLIT_POINTS = [5,10,15,21]  # 15 + 6 gens , 5 window_size , 5 for number of unit1 and unit2   [4,9,14]
                                                                                        # [31, 31, 31 , 63]

                                                                                     # Last overall number of units   27


arguments = sys.argv
input_chromosome = arguments[1]


# print("===================")
# print(input_chromosome)
# print("===================")

def main():
    global DATA_PATH
    global SPLIT_POINTS
    global input_chromosome

    window_size, num_units_1, num_units_2, num_dense1, num_dense2 = Bin_to_Dec(input_chromosome, SPLIT_POINTS)
    data = data_import(DATA_PATH)
    test_acc = model(data, window_size, num_units_1, num_units_2, num_dense1, num_dense2)
    return test_acc
    

if __name__ == '__main__':    
    sys.stdout.write(str(main()))
