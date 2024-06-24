from abc import ABC, abstractmethod
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import dimod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import dimod
from tqdm import tqdm
from random import randint
from tqdm import trange
from numpy.linalg import norm
from matplotlib import pyplot as plt
from minorminer import find_embedding
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from neal import SimulatedAnnealingSampler
import dimod
import math
from dwave.system import DWaveSampler
from tqdm import tqdm
from pathlib import Path
from enum import Enum
from tqdm import trange
import time

API_KEY = "THE API KEY FOR ACCESSING QUANTUM ANNEALERS"

class SubsetExtractionData:
    def __init__(self, label="SCos-QUBO"):

        # The number of samples
        self.number_of_samples=0
        # The network/enqueueing time considering annealing
        self.network_enqueueing_time=0
        # The total annealing time
        self.total_annealing_time=0
        # The first matrix building time
        self.total_first_matrix_time=0
        # The total amount of time perceived by the user
        self.total_approach_effective_time=0
        # The total time to find the embedding
        self.total_embedding_time = 0
        # The total time to formulate the problems in BQM/QUBO
        self.total_formulation_time=0
        # The label of the approach
        self.label = label

    def add_network_time(self, t):
        self.network_enqueueing_time+=t

    def add_annealing_time(self, t):
        self.total_annealing_time+=t
        self.number_of_samples+=1

    def add_first_matrix_time(self, t):
        self.total_first_matrix_time+=t
    
    def add_total_approach_time(self, t):
        self.total_approach_effective_time+=t

    def add_embedding_time(self, t):
        self.total_embedding_time+=t

    def add_formulation_time(self, t):
        self.total_formulation_time+=t

    def print_data(self,):
        print()
        print(f"####### Subset Extraction Timings for {self.label}: #######")

        average_qpu_access_time = self.total_annealing_time / max(1, self.number_of_samples)
        average_qpu_latency = (self.network_enqueueing_time-self.total_annealing_time )/ max(1, self.number_of_samples)

        print(f"Total Time required by this approach= {round((self.total_approach_effective_time)/1000, 2)} ms")
        print(f"Total Time required for building first matrix= {round(self.total_first_matrix_time/1000, 2)} ms")
        print(f"Total Embedding Time= {round(self.total_embedding_time/1000, 2)} ms (*)")
        print(f"Total QUBO/BQM Formulation Time= {round(self.total_formulation_time/1000, 2)} ms")
        # Obs: in case of multicore, this does not represent the time seen by the developer but rather
        # the total time that has been split among several cores. Therefore, it is likely that
        # the developer sees a lower amount of time but several cores performed Annealing processes.
        print(f"Total Annealing (QPU Access Time) Time= {round(self.total_annealing_time/1000, 2)} ms\t[Average= {round(average_qpu_access_time/1000, 2)} ms over {self.number_of_samples} samples]")
        print(f"Total Network latencies and Enqueueing Time= {round((self.network_enqueueing_time -self.total_annealing_time)/1000, 2)} ms\t[Average= {round((average_qpu_latency)/1000, 2)} ms over {self.number_of_samples} samples] (*)")
        print()
        print("(*) This parameter should be skept if considering Simulated Annealing")
        print()


class SAMPLER_TYPE(Enum):
    QUANTUM = 1
    SIMULATED = 2


class QuantumBatch:
    def __init__(self, data, embedding=None, label="", docs_range=[]):
        self.data = data
        self.embedding = embedding
        self.label = label
        self.docs_range = docs_range
        self.sampler=None
        self.bqm=None


class QuantumResponse:
    def __init__(self, response, batch):
        self.response = response
        self.batch = batch

    def is_response_done(self,):
        return self.response.done()
    
    def get_kept_docs_async(self,):
        if(not self.is_response_done()):
            return None
        final_response={}
        for var, index in zip(self.batch.docs_range, sorted(self.response.first.sample.keys())):
            final_response[var] = self.response.first.sample[index]
        return final_response
    
    def get_kept_docs_sync(self,):
        final_response={}
        for var, index in zip(self.batch.docs_range, sorted(self.response.first.sample.keys())):
            final_response[var] = self.response.first.sample[index]
        return final_response
    
    def get_response_qpu_access_time_async(self,):
        if(not self.is_response_done()):
            return None
        return self.response.info['timing']['qpu_access_time']
    
    def get_response_qpu_access_time_sync(self,):
        return self.response.info['timing']['qpu_access_time']
    

class GeneralQuboSolver(ABC):
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.X_train_tfidf=X_train
        """
        if(X_val is None):
            self.X_train_tfidf=self.tfidf_vectorize(X_train).toarray()
        else:
            train,val=self.tfidf_vectorize_train_val(X_train,X_val)
            self.X_train_tfidf=train.toarray() 
            self.X_val_tfidf=val.toarray()"""
        self.responses=[]
        
        self.subset_extraction_data=SubsetExtractionData(label="SCos-QUBO")

    @abstractmethod
    def build_bqm(self, batch:QuantumBatch, percentage):#, features, target, docs=None):
        """Builds the Binary Quadratic Model that represents the problem
        of interest.
        
        :param dataset: the dataset where the last column represents the scores of the weak models
        :param features: the list of documents (columns) of the dataset
        :param target: the column representing the scores of the weak models

        """
        pass

    @abstractmethod
    def prepare_initial_matrix(self, cores=1):
        """Prepares the initial matrix that will be converted to QUBO and BQM later.

        """
        pass
        
    @abstractmethod
    def extract_subset(self,):
        """Extracts the subset of the documents that should be kept in the training set according
        to the method used by this class.

        """
        pass

    def compute_embedding(self, batch, percentage):
        """Computes the embedding that can be reused later.

        :param batch: the initial matrix from which to derive the QUBO/BQM formulation
        :param percentage: the percentage of instances to be kept

        """
        print("Computing embedding...")
        
        kbqm = self.build_bqm(batch,percentage)
        qubo = kbqm.to_qubo()[0]
        solver = DWaveSampler(token=API_KEY,solver="Advantage_system4.1")
        __, target_edgelist, _ = solver.structure
        emb = find_embedding(qubo, target_edgelist, verbose=1)
        
        print("Finished computing embedding.")
        return emb
    
    def get_responses(self,):
        """
        Processes the responses that are being received in an async way.
        This loops through the responses and check whether their status
        is finished or not. If finished, then it retrieves the solution.
        This spares a huge amount of time!
        """
        docs={}
        with tqdm(total=len(self.responses),desc=f"Waiting for responses...") as pbar:
            while(len(self.responses)>0):
                to_remove=None
                for i,response in enumerate(self.responses):
                    to_remove=None
                    docs_to_keep=response.get_kept_docs_async()
                    if(docs_to_keep is not None):
                        self.subset_extraction_data.add_annealing_time(response.get_response_qpu_access_time_async())
                        for key in docs_to_keep.keys():
                            docs[key]=docs_to_keep[key]
                        to_remove=i
                        break
                if(to_remove is not None):
                    del self.responses[to_remove]
                    pbar.update(1)
        return docs
    
    def get_best_instances_quantum(self, batch: QuantumBatch, percentage=0.8, num_reads=100):
        """
        Samples from the batch using the quantum annealer and appends
        the response to an inner list of responses, waiting for their
        status to be finished and then processed.
        """
        sampler = batch.sampler
        to_keep = int((batch.data.shape[0])*percentage)

        response, network_delay = self.extract_top_k_documents(sampler, batch, to_keep, num_reads)
        
        return response, network_delay

    def get_best_instances_multiprocess_sa(self, batch: QuantumBatch, percentage=0.8, num_reads=100):
        """
        Samples from the batch using the Simulated Annealing algorithm 
        and returns immediately the response. This is done synchronously
        since there is no network involved and we can exploit parallelization
        to speed up this operation.
        """
        sampler = SimulatedAnnealingSampler()
        to_keep = int((batch.data.shape[0])*percentage)
        
        response, _ = self.extract_top_k_documents(sampler, batch, to_keep, num_reads)
        response_time=response.info['timing']['preprocessing_ns']/1000+response.info['timing']['sampling_ns']/1000+response.info['timing']['postprocessing_ns']/1000

        final_response = {}
        final_response['annealing_time']=response_time
        for var, index in zip(batch.docs_range, sorted(response.first.sample.keys())):
            final_response[var] = response.first.sample[index]
        
        return final_response


    def extract_top_k_documents(self, sampler, batch, k, num_reads):
        """
        Formulates the problem according to the batch and k provided and
        then samples it.
        """
        kbqm = batch.bqm
        start_time = time.perf_counter()
        response = sampler.sample(kbqm, label=batch.label, num_reads=num_reads)
        end_time = time.perf_counter()
        network_time=(end_time - start_time) * 1e6
        return response,network_time


    def split_in_batches(self, batch_size, sampler_type, percentage_kept, start_index=0):
        """Splits the provided initial matrix into batches having size that can be at max
        the specified one.

        :param matrix: the initial matrix from which to derive the QUBO/BQM formulation
        :param batch_size: the maximum batch size
        :sampler_type: the type of sampler to use (e.g., Quantum or Simulated)
        :param percentage_kept: the percentage of instances to be kept
        :start_index: the starting index, useful when computing many matrices (e.g., in SmartHamQuboSolver)

        """
        batches = []

        # The batches having one more samples if the division is not integer
        number_one_more = self.X_train_tfidf.shape[0] % math.ceil(
            self.X_train_tfidf.shape[0]/batch_size)
        
        matrix_splitted = np.array_split(
            self.X_train_tfidf, math.ceil(self.X_train_tfidf.shape[0]/batch_size), axis=0)

        range_start = start_index
        range_end = start_index
        embedding_time=0
        embedding_computed = None
        
        for i in range(number_one_more):
            range_end += (matrix_splitted[i].shape[0])
            batch=QuantumBatch(
                data=matrix_splitted[i],
                embedding=embedding_computed,
                label=f"QA-IS -> Batch={i}",
                docs_range=np.arange(range_start, range_end)
            )
            if (embedding_computed is None and sampler_type == SAMPLER_TYPE.QUANTUM):
                
                start_time = time.perf_counter()
                embedding_computed = self.compute_embedding(
                    batch, percentage_kept)
                end_time = time.perf_counter()
                embedding_time+=(end_time - start_time) * 1e6
                batch.embedding=embedding_computed
            batches.append(batch)
            range_start += (matrix_splitted[i].shape[0])

        embedding_computed = None
        for i in range(number_one_more, len(matrix_splitted)):
            range_end += (matrix_splitted[i].shape[0])
            batch=QuantumBatch(
                data=matrix_splitted[i],
                embedding=embedding_computed,
                label=f"QA-IS -> Batch={i}",
                docs_range=np.arange(range_start, range_end)
            )
            if (embedding_computed is None and sampler_type == SAMPLER_TYPE.QUANTUM):
                
                start_time = time.perf_counter()
                embedding_computed = self.compute_embedding(
                    batch, percentage_kept)
                end_time = time.perf_counter()
                embedding_time+=(end_time - start_time) * 1e6
                batch.embedding=embedding_computed
            
            batches.append(batch)
            range_start += (matrix_splitted[i].shape[0])
        
        return batches, embedding_time

    def sample_train_set_weak(self, X, y):
        percentage = self.get_percentage_train_weak(X)
        X_train_classifier, _, y_train_classifier, _ = train_test_split(
            np.array(X), np.array(y), stratify=y, train_size=percentage)
        return X_train_classifier, y_train_classifier
    

    def extract_training_subset(self,subset_dict):
        indexes = []
        for k in sorted(subset_dict.keys()):
            if (subset_dict[k] > 0.5):
                indexes.append(k)
        
        return indexes
    
    def maximum_energy_delta(self, bqm):
        """Computes conservative bound on maximum change in energy when flipping a single variable"""
        return max(abs(bqm.get_linear(i))
                   + sum(abs(bqm.get_quadratic(i, j))
                         for j in (v for v, _ in bqm.iter_neighborhood(i)))
                   for i in iter(bqm.variables))

    def print_response_data(self, response):
        # ------- Print results to user -------
        print('-' * 65)
        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(
            'Set 0', 'Set 1', 'Energy', "Count"))
        print('-' * 65)
        for sample, E, occ in response.data(fields=['sample', 'energy', "num_occurrences"]):
            S0 = [k for k, v in sample.items() if v == 0]
            S1 = [k for k, v in sample.items() if v == 1]
            print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(
                str(S0), str(S1), str(E), str(occ)))

    def plot_qubo_matrix(self, bqm, features):
        Q_np = np.ones((len(features), len(features)))*np.nan

        feature_to_index = {feature: index for index,
                            feature in enumerate(features)}
        
        for (i, j), value in bqm.to_qubo()[0].items():

            Q_np[i, j] = value
        plt.imshow(Q_np, cmap='jet')
        plt.colorbar()
        plt.show()

    def plot_qubo_matrix_1(self, qubo, features):
        Q_np = np.ones((len(features), len(features)))*np.nan
        for (i, j), value in qubo.items():
            Q_np[i, j] = value

        #Q_np=np.tril(Q_np)
        #Q_np[np.triu_indices(Q_np.shape[0], 1)] = np.nan
        
        plt.imshow(Q_np, cmap='jet')
        plt.colorbar()
        plt.show()

    def print_timing_data(self,):
        self.subset_extraction_data.print_data()

    def tfidf_vectorize(self,dataset):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(dataset)
        return X

    def tfidf_vectorize_train_val(self,X_train, X_val):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=2000)
        X=np.concatenate((X_train,X_val))
        vectorizer.fit(X)
        X_t = vectorizer.transform(X_train)
        X_v = vectorizer.transform(X_val)
        return X_t.toarray(), X_v.toarray()