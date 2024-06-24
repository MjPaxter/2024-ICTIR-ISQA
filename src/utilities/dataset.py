from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class Dataset(ABC):
    def __init__(self,):
        self.path=None

    def get_dataset_train_validation_test(self):
        return ((self.X_train,self.y_train),(self.X_validation,self.y_validation),(self.X_test,self.y_test))

    def read_dataset_from_path(self, path, X_col_name, y_col_name, frac=1):
        dataset = pd.read_csv(path)

        le=LabelEncoder().fit(dataset[y_col_name].values.tolist())

        dataset[y_col_name] = le.transform(dataset[y_col_name])
        dataset=dataset.dropna()
        dataset=dataset.sample(frac=frac)

        return np.array(dataset[X_col_name]),np.array(dataset[y_col_name])
    

class GeneralDataset(Dataset):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_validation=X_val
        self.y_validation=y_val
        self.X_test=X_test
        self.y_test=y_test
    

class Agnews(Dataset):
    def __init__(self, fold):
        self.path= Path(os.getcwd()) / "datasets" / "AGNEWS"
        
        self.X_train_col_name="Description"
        self.y_train_col_name="Class Index"
        self.X_validation_col_name="Description"
        self.y_validation_col_name="Class Index"
        self.X_test_col_name="Description"
        self.y_test_col_name="Class Index"

        train_path=self.path / f"fold_{fold}" / "train.csv"
        validation_path=self.path / f"fold_{fold}" / "validation.csv"
        test_path=self.path / f"fold_{fold}" / "test.csv"

        self.X_train,self.y_train=self.read_dataset_from_path(train_path, self.X_train_col_name,self.y_train_col_name)
        self.X_validation,self.y_validation=self.read_dataset_from_path(validation_path, self.X_validation_col_name,self.y_validation_col_name)
        self.X_test,self.y_test=self.read_dataset_from_path(test_path, self.X_test_col_name,self.y_test_col_name)
