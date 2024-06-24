import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neural_network import MLPClassifier
import pandas as pd
from src.utilities import helper_functions as f
from random import randint
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utilities.general_solver import SAMPLER_TYPE
import pickle
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utilities import dataset as d
import sys
np.random.seed(0)


def get_splits(splits_filename):
	with open(splits_filename, "rb") as splits_file:
		return pickle.load(splits_file)

def readfile(filename):
	with io.open(filename, 'rt', newline='\n', encoding='utf8', errors='ignore') as filein:
		return filein.readlines()

def get_array(X, idxs):
    return [X[idx] for idx in idxs]

def get_splits_by_fold(X, y, df_splits, f):

	train_idx = df_splits.loc[f].train_idxs
	test_idx = df_splits.loc[f].test_idxs
	val_idx = df_splits.loc[f].val_idxs

	X_train, y_train = get_array(X, train_idx), get_array(y, train_idx)
	X_test, y_test = get_array(X, test_idx), get_array(y, test_idx)
	X_val, y_val = get_array(X, val_idx), get_array(y, val_idx)

	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_val), np.array(y_val)

def scan_directory(dirname):
    directories = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    return directories



if __name__== "__main__":

    ###### CONFIGURATION #################
    sampler_type = SAMPLER_TYPE.SIMULATED
    method = f.METHOD_TYPE.SMART_COS
    percentage_kept = 0.75
    cores = 4
    batch_size = 80
    num_reads = 200 #50 reads is 25ms approximately, and then every 50 reads is 5 ms more. 200 seems to be a good number
    ######################################

    folder="yelp_reviews_2L"
    df_splits = get_splits(f"..\datasetsForAndrea\{folder}\splits\split_5_with_val.pkl")
    y = LabelEncoder().fit_transform(np.array(list(map(int, readfile(f'..\datasetsForAndrea\{folder}\score.txt')))))
    X = readfile(f'..\\datasetsForAndrea\\{folder}\\texts.txt')
    
    #orig_stdout = sys.stdout

    for fold in range(0,4):

        if not os.path.isdir(f"results_qa\{folder}"): 
            os.makedirs(f"results_qa\{folder}")
        
        X_train, y_train, X_test, y_test, X_val, y_val = get_splits_by_fold(X, y, df_splits, fold)
        idx = np.random.permutation(len(y_train))
        X_train=X_train[idx]
        y_train=y_train[idx]

        dataset=d.GeneralDataset(X_train, y_train, X_test, y_test, X_val, y_val)
    
        X_train_subset,y_train_subset,_=f.extract_subset(dataset, method, sampler_type, batch_size, percentage_kept, cores, num_reads)