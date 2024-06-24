import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from random import randint
from sklearn.datasets import load_svmlight_file
import numpy as np
from pathlib import Path
from enum import Enum
from tqdm import trange
from matplotlib import pyplot as plt
from minorminer import find_embedding
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from src.utilities.general_solver import SAMPLER_TYPE
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import sys
from .dataset import Dataset


class METHOD_TYPE(Enum):
    MUTUAL_INFORMATION = 1
    SMART_COS = 2
    SMART_EUCLIDEAN = 3


def read_dataset_train_test(train_path, test_path, X_train_col, y_train_col, X_test_col, y_test_col):
    train_dataset = pd.read_csv(train_path)

    le=LabelEncoder().fit(train_dataset[y_train_col].values.tolist())

    train_dataset[y_train_col] = le.transform(train_dataset[y_train_col])
    train_dataset=train_dataset.dropna()

    test_dataset = pd.read_csv(test_path)

    test_dataset[y_test_col] = le.transform(test_dataset[y_test_col])
    test_dataset=test_dataset.dropna()

    return np.array(train_dataset[X_train_col]),np.array(train_dataset[y_train_col]), np.array(test_dataset[X_test_col]),np.array(test_dataset[y_test_col])


def read_dataset_washington(inputdir, f):
    X_train, y_train = load_svmlight_file(
        inputdir+"train"+str(f)+".gz", dtype=np.float64)
    X_test, y_test = load_svmlight_file(
        inputdir+"test"+str(f)+".gz", dtype=np.float64)
    X_val, y_val = load_svmlight_file(
        inputdir+"val"+str(f)+".gz", dtype=np.float64)

    n_features = max(X_train.shape[1], X_test.shape[1], X_val.shape[1])

    X_train, y_train = load_svmlight_file(
        inputdir+"train"+str(f)+".gz", dtype=np.float64, n_features=n_features)
    X_test, y_test = load_svmlight_file(
        inputdir+"test"+str(f)+".gz", dtype=np.float64, n_features=n_features)
    X_val, y_val = load_svmlight_file(
        inputdir+"val"+str(f)+".gz", dtype=np.float64, n_features=n_features)

    return X_train.toarray(), y_train, X_val.toarray(), y_val, X_test.toarray(), y_test


def extract_subset(dataset:Dataset, method: METHOD_TYPE, sampler_type: SAMPLER_TYPE,  batch_size: int, percentage_kept: float, cores: int, num_reads: float):
    solver = None
    
    """if (method == METHOD_TYPE.MUTUAL_INFORMATION):
        import mutual_information_qubo_solver as q
        solver = q.MutualInformationQuboSolver(
            X_train_tfidf, y_train, X_val_tfidf, y_val)"""

    if (method == METHOD_TYPE.SMART_COS):
        from . import scos_solver as q
        solver = q.SmartCosineQuboSolver(
            dataset)

    if (method == METHOD_TYPE.SMART_EUCLIDEAN):
        from . import seu_solver as q
        solver = q.SmartEuclideanQuboSolver(
            dataset)

    subset = solver.extract_subset(
        batch_size, sampler_type, percentage_kept, cores, num_reads)
    
    indexes = solver.extract_training_subset(subset)

    solver.print_timing_data()

    # TODO: REMOVE SUBSET!!!
    return dataset.X_train[indexes], dataset.y_train[indexes], subset


def train_validation_test_split(X, y, sizes=[0.7, 0.1, 0.2]):
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, np.arange(X.shape[0]), train_size=sizes[0])
    val_size = (normalize([sizes[1:]], norm="l1")[0])[0]
    X_val, X_test, y_val, y_test, _, _ = train_test_split(
        X_test, y_test, np.arange(X_test.shape[0]), train_size=val_size)

    return X_train, y_train, X_val, y_val, X_test, y_test,


def tfidf_vectorize(X_train, X_subset, X_val, X_test):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=2000)
        #X=np.concatenate((X_train,X_val,X_test,X_subset))
        vectorizer.fit(X_train)
        X_tr = vectorizer.transform(X_train)
        X_v = vectorizer.transform(X_val)
        X_s = vectorizer.transform(X_subset)
        X_te = vectorizer.transform(X_test)
        return X_tr.toarray(), X_s.toarray(), X_v.toarray(), X_te.toarray()



def process_washington_dataset(dir, method, sampler_type: SAMPLER_TYPE,  batch_size: int, percentage_kept: float, cores: int, num_reads: float):
    run_data = RunData(label_new_method="Smart")

    for t in range(10):
        print(f"PROCESSING DATASET NUMBER {t+1}")
        X_train, y_train, X_val, y_val, X_test, y_test, = read_dataset_washington(
            dir, t)
        X_train_subset, y_train_subset = extract_subset(
            X_train, X_train, y_train, X_val, y_val, method, sampler_type, batch_size, percentage_kept, cores, num_reads)

        measure_effectiveness_svm(X_train, y_train, X_train_subset,
                                  y_train_subset, X_test, y_test, percentage_kept, run_data)
    original_stdout = sys.stdout
    with open("scos_washington_svm.txt", "a") as f:
        sys.stdout = f
        print(dir)
        run_data.print_data()
        sys.stdout = original_stdout


def measure_effectiveness_nn(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, percentage_kept, run_data):
    preds = make_pipeline(StandardScaler(),MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(
        200, 150, 120, 80, 50, 25, 10), early_stopping=True)).fit(X_train_subset, y_train_subset).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.new_method_data.append(TrainData(score*100, 0))

    X_random, _, y_random, _ = train_test_split(
        X_train, y_train, train_size=percentage_kept)
    preds = make_pipeline(StandardScaler(),MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(
        200, 150, 120, 80, 50, 25, 10), early_stopping=True)).fit(X_random, y_random).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.random_data.append(TrainData(score*100, 0))

    preds = make_pipeline(StandardScaler(),MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(
        200, 150, 120, 80, 50, 25, 10), early_stopping=True)).fit(X_train, y_train).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.full_data.append(TrainData(score*100, 0))

def measure_effectiveness_rfc(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, percentage_kept, run_data):
    from sklearn.ensemble import RandomForestClassifier
    preds = RandomForestClassifier(max_depth=30, random_state=0).fit(X_train_subset, y_train_subset).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.new_method_data.append(TrainData(score*100, 0))

    X_random, _, y_random, _ = train_test_split(
        X_train, y_train, train_size=percentage_kept)
    preds = RandomForestClassifier(max_depth=30, random_state=0).fit(X_random, y_random).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.random_data.append(TrainData(score*100, 0))

    preds = RandomForestClassifier(max_depth=30, random_state=0).fit(X_train, y_train).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.full_data.append(TrainData(score*100, 0))


def measure_effectiveness_svm(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, percentage_kept, run_data):
    preds = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=0)).fit(
        X_train_subset, y_train_subset).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.new_method_data.append(TrainData(score*100, 0))

    X_random, _, y_random, _ = train_test_split(
        X_train, y_train, train_size=percentage_kept,random_state=0)
    preds = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=0)).fit(
        X_random, y_random).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.random_data.append(TrainData(score*100, 0))

    preds = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=0)).fit(
        X_train, y_train).predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    run_data.full_data.append(TrainData(score*100, 0))



class TrainData:
    def __init__(self, score, time):
        self.score = score
        self.time = time


class RunData:
    def __init__(self, label_new_method="Smart"):
        self.new_method_data = []
        self.random_data = []
        self.full_data = []
        self.label_new_method = label_new_method

    def print_data(self,):

        print(f"####### Data from {self.label_new_method}: #######")
        for e in self.new_method_data:
            print(f"Score= {round(e.score, 4)}\tTime= {round(e.time, 4)}")
        average_score = sum(
            [e.score for e in self.new_method_data]) / max(1, len(self.new_method_data))
        average_time = sum(
            [e.time for e in self.new_method_data]) / max(1, len(self.new_method_data))
        if (len(self.new_method_data) > 1):
            variance_score = statistics.variance(
                [e.score for e in self.new_method_data])
            print(
                f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {round(variance_score, 4)}")
        else:
            variance_score = "Undefined"
            print(
                f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {variance_score}")

        print("\n")
        print(f"####### Data from Random: #######")
        for e in self.random_data:
            print(f"Score= {round(e.score, 4)}\tTime= {round(e.time, 4)}")
        average_score = sum(
            [e.score for e in self.random_data]) / max(1, len(self.random_data))
        average_time = sum([e.time for e in self.random_data]
                           ) / max(1, len(self.random_data))
        if (len(self.random_data) > 1):
            variance_score = statistics.variance(
                [e.score for e in self.random_data])
            print(
                f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {round(variance_score, 4)}")
        else:
            variance_score = "Undefined"
            print(
                f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {variance_score}")

        if (len(self.full_data) > 0):
            print("\n")
            print(f"####### Data from Full: #######")
            for e in self.full_data:
                print(f"Score= {round(e.score, 4)}\tTime= {round(e.time, 4)}")
            average_score = sum(
                [e.score for e in self.full_data]) / max(1, len(self.full_data))
            average_time = sum(
                [e.time for e in self.full_data]) / max(1, len(self.full_data))
            if (len(self.full_data) > 1):
                variance_score = statistics.variance(
                    [e.score for e in self.full_data])
                print(
                    f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {round(variance_score, 4)}")
            else:
                variance_score = "Undefined"
                print(
                    f"Average Score= {round(average_score, 4)}\tAverage Time= {round(average_time, 4)}\tVariance Score= {variance_score}")
