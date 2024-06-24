import src.utilities.general_solver as q
import pandas as pd
import numpy as np
from tqdm import trange
from tqdm import tqdm
import dimod,math
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import copy
import contextlib
import joblib
from joblib import Parallel, delayed
import time
from functools import wraps
from src.utilities.dataset import Dataset
#from general_solver import SAMPLER_TYPE,API_KEY, QuantumResponse
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Shows tqdm progress bar when using joblib
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class SmartEuclideanQuboSolver(q.GeneralQuboSolver):
    """
    Solver using the SCos-QUBO method to extract the best
    subset of samples from a given dataset.
    Inherits from GeneralQuboSolver.
    """

    def __init__(self, dataset:Dataset):
        """
        Initializes the solver

        Parameters:
        - dataset (Dataset): The dataset considered.
        """

        (X_train,y_train),_,_=dataset.get_dataset_train_validation_test()
        super().__init__(X_train, y_train, None, None)


    def __eucli_between_2_variables(self,x1,x2,y1,y2):
        """
        Computes 1/euclidean distance between 2 vectors according
        to their labels. If label(x1)==label(x2) returns |cosine| else
        returns -|cosine|.

        Parameters:
        - x1 (array): The first vector.
        - x2 (array): The second vector.
        - y1 (int): The label associated to the first vector.
        - y2 (int): The label associated to the second vector.

        Returns:
        int: '1/euclidean deistance' between the 2 vectors
        """

        if norm(x1) == 0 or norm(x2) == 0:
            return 0.0
        distance=1/(0.001+math.sqrt(np.linalg.norm(x1-x2)))
        
        if(y1!=y2):
            distance=-distance
        return distance
    
    def normalize_dict_values(self,dictionary):
        # Find the minimum and maximum values in the dictionary
        min_val = min(dictionary.values())
        max_val = max(dictionary.values())
        normalized_dict={}
        for key, value in dictionary.items():
            if(value>=0):
                normalized_dict[key]=-value/ (max_val)
            else:
                normalized_dict[key]=1*value/ (min_val)
        
        return normalized_dict


    def __class_balance(self, label_x1):
        """
        Computes the balancing factor of the label of the class.

        Parameters:
        - label_x1 (int): The label of the class.

        Returns:
        int: The balancing factor for the classs
        """

        coefficient = (self.y_train == label_x1).sum()/self.y_train.shape[0]
        
        
        return coefficient


    def build_bqm(self, batch, percentage):
        """
        Builds the bqm for the batch according to the specific functions that are 
        used in this approach.

        Parameters:
        - batch (array): The batch.
        - percentage (float): The percentage of documents to keep in this batch.

        Returns:
        BinaryQuadraticModel: The bqm.
        """

        qubo = {}
        
        for i, _ in enumerate(batch.data):
            for j, _ in enumerate(batch.data):
                if (i != j):
                    qubo[(i, j)] = self.__eucli_between_2_variables(
                        self.X_train_tfidf[i+batch.docs_range[0]], self.X_train_tfidf[j+batch.docs_range[0]],
                        self.y_train[i+batch.docs_range[0]], self.y_train[j+batch.docs_range[0]]
                        )
        qubo=self.normalize_dict_values(qubo)
        for i, _ in enumerate(batch.data):
            qubo[(i, i)] = self.__class_balance(
                self.y_train[i+batch.docs_range[0]]
                )
        
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=0.0)

        k = int((batch.data.shape[0])*(percentage))
        penalty = self.maximum_energy_delta(bqm)
        kbqm = dimod.generators.combinations(
            bqm.variables, k, strength=penalty)
        kbqm.update(bqm)
        #features = batch.docs_range
        #self.plot_qubo_matrix_1(qubo,features)
        #features = batch.docs_range
        #self.plot_qubo_matrix(bqm,features)

        return bqm
    
    def prepare_initial_matrix(self, cores=1):
        raise Exception("NOT IMPLEMENTED")
    
    
    def quantum_sample(self,batches,percentage_kept,num_reads):
        """
        Samples from the provided batches using the quantum annealer.

        Parameters:
        - batches (array): The batches.
        - percentage (float): The percentage of documents to keep.
        - num_reads (int): The number of reads for each submitted problem.
        """

        sampler=None
        embedding=batches[0].embedding
        with tqdm(total=len(batches),desc =f"Sampling using 1 core...") as pbar:
            for batch in batches:
                network_delay=0
                if(batch.embedding==embedding):
                    # Starts measuring the time required for statistics
                    start_time=time.perf_counter()
                    if(sampler is None):
                        sampler = FixedEmbeddingComposite(
                            DWaveSampler(token=q.API_KEY,solver="Advantage_system4.1"), batch.embedding)
                    # Finishes measuring the time required for statistics
                    end_time=time.perf_counter()
                    network_delay=(end_time-start_time)*1e6
                    batch.sampler=sampler
                    response,network_time=self.get_best_instances_quantum(batch,percentage_kept,num_reads)
                    self.responses.append(q.QuantumResponse(response,batch))
                    self.subset_extraction_data.add_network_time(network_time)
                    self.subset_extraction_data.add_network_time(network_delay)
                    
                else:
                    embedding=batch.embedding
                    # Starts measuring the time required for statistics
                    start_time=time.perf_counter()
                    sampler = FixedEmbeddingComposite(
                            DWaveSampler(token=q.API_KEY,solver="Advantage_system4.1"), batch.embedding)
                    # Finishes measuring the time required for statistics
                    end_time=time.perf_counter()
                    network_delay=(end_time-start_time)*1e6
                    batch.sampler=sampler
                    response,network_time=self.get_best_instances_quantum(batch,percentage_kept,num_reads)
                    self.responses.append(q.QuantumResponse(response,batch))
                    self.subset_extraction_data.add_network_time(network_time)
                    self.subset_extraction_data.add_network_time(network_delay)
                
                pbar.update(1)


    def extract_subset(self, batch_size, sampler_type, percentage_kept, cores, num_reads):
        """
        Produces the subset according to this approach of Instance Selection.
        It is the main function that calls all the others as utilities to compute
        the final subset of documents to keep.

        Parameters:
        - batch_size (int): The batch size.
        - sampler_type (enum): The type of sampler used (Simulated Annealing or Quantum Annealing).
        - cores (int): The number of cores.
        - num_reads (int): The number of reads for each submitted problem.
        - percentage_kept (float): The percentage of documents to keep in this batch.

        Returns:
        dict: The documents to keep as a dictionary where each key
        represents the index of the document in the original dataset.
        Each key is associated with a 0 (remove) and 1 (keep).
        """

        # Starts measuring the time required for statistics
        start_time_extraction = time.perf_counter()
        #matrices = self.prepare_initial_matrix(cores)
        docs = {}
        if(sampler_type==q.SAMPLER_TYPE.QUANTUM):
            
            # Splits each matrix in batches and calculates embedding if using Quantum Annealing
            batches,embedding_time = self.split_in_batches(
                    batch_size, sampler_type, percentage_kept, start_index=0)
            self.subset_extraction_data.add_embedding_time(embedding_time)

            start_time=time.perf_counter()
            bqms=None
            with tqdm_joblib(tqdm(desc=f"Formulating problems using {cores} cores...", total=len(batches))) as progress_bar:
                bqms = Parallel(n_jobs=cores)(delayed(self.build_bqm)(
                        batch,
                        percentage_kept
                    ) for batch in batches)
            
            for batch,bqm in zip(batches,bqms):
                batch.bqm=bqm
            end_time=time.perf_counter()
            formulation_time=(end_time-start_time)*1e6
            self.subset_extraction_data.add_formulation_time(formulation_time)
            
            self.quantum_sample(batches,percentage_kept,num_reads)
            
            # Starts measuring the time required for statistics
            start_time=time.perf_counter()
            docs=self.get_responses()
            # Finishes measuring the time required for statistics
            end_time=time.perf_counter()
            network_delay=(end_time-start_time)*1e6
            self.subset_extraction_data.add_network_time(network_delay)

        else:
            # Splits each matrix in batches and calculates embedding if using Quantum Annealing
            batches,_ = self.split_in_batches(
                batch_size, sampler_type, percentage_kept, start_index=0)
            
            start_time=time.perf_counter()
            bqms=None
            with tqdm_joblib(tqdm(desc=f"Formulating problems using {cores} cores...", total=len(batches))) as progress_bar:
                bqms = Parallel(n_jobs=cores)(delayed(self.build_bqm)(
                        batch,
                        percentage_kept
                    ) for batch in batches)
            
            for batch,bqm in zip(batches,bqms):
                batch.bqm=bqm
            end_time=time.perf_counter()
            formulation_time=(end_time-start_time)*1e6
            self.subset_extraction_data.add_formulation_time(formulation_time)

            with tqdm_joblib(tqdm(desc=f"Sampling using {cores} cores...", total=len(batches))) as progress_bar:
                results = Parallel(n_jobs=cores)(delayed(self.get_best_instances_multiprocess_sa)(
                    batch,
                    percentage=percentage_kept,
                    num_reads=num_reads,
                ) for batch in batches)
                for sub in results:
                    self.subset_extraction_data.add_annealing_time(sub['annealing_time'])
                    del sub['annealing_time']
                    
                    for i in sub.keys():
                        docs[i] = sub[i]
            
        zeros = 0
        for i in docs.keys():
            if (docs[i] < 0.5):
                zeros += 1
        
        print(f"Over {len(docs.keys())} samples we have {len(docs.keys())-zeros}/{len(docs.keys())} kept ({100*(len(docs.keys())-zeros)/len(docs.keys())}%). Dataset size is {self.X_train_tfidf.shape[0]}")
    
        # Finishes measuring the time required for statistics
        end_time_extraction = time.perf_counter()
        self.subset_extraction_data.total_approach_effective_time+=(end_time_extraction - start_time_extraction) * 1e6

        return docs