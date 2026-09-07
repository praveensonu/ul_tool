from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from ctypes import c_bool, c_int
from .data_loader import get_model_tokenizer, TrainDataset, TestDataset, get_tokenizer, get_model
from .calc_inner import grad_z
from .utils import save_json, display_progress, load_json
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from copy import copy
from pathlib import Path
import torch
import time
import math
import pickle

MAX_DATASET_SIZE = int(1e8)


class RapidGrad():
    def __init__(self, config, map_location, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.shuffle_lambda = config.influence.RapidGrad.shuffle_lambda*2
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.config = config
        self.map_location = map_location
        self.seed = seed

    def __call__(self, vec, K):
        # Apply identical permutations/projections to each sample independently.
        single = vec.ndim == 1
        vec = vec.unsqueeze(0) if single else vec
        if vec.ndim != 2:
            raise ValueError("RapidGrad expects a vector or a batch of vectors.")
        batch_size, dimension = vec.shape
        if not self.is_init:
            print("Creating random and shuffling matrices. It may take a few minutes.")
            self.init(dimension)
        if dimension != self.D:
            raise ValueError("Gradient dimension changed within a RapidGrad run.")
        for i, (dim, permutation) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i % 2 == 0:
                vec = vec.reshape(batch_size, dim, -1)[:, permutation, :]
            else:
                vec = vec.reshape(batch_size, -1, dim)[:, :, permutation]
        vec = vec.reshape(batch_size, dimension) * self.random_mat

        def project(k):
            projected = vec.reshape(batch_size, k, self.D // k).sum(dim=2)
            return projected[0] if single else projected

        return [project(k) for k in K] if isinstance(K, list) else project(K)

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        self.D = D
        self.file_name = os.path.join(
            self.config.influence.grads_path,
            f"RapidGrad_D{self.D}_n{self.shuffle_lambda}.obj"
        )
        if not self.load():
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()
        self.random_mat = torch.from_numpy(self.random_mat).to(dtype=torch.float16).to(self.map_location)

    def create_random_mat(self, D):
        self.random_mat = np.random.randint(0, 2, (D,), dtype=np.int8)
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        lt = []
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D / i
                    break
        for _ in tqdm(range(self.shuffle_lambda)):
            x = np.random.randint(len(lt)//4, len(lt)//2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

    def save(self):
        if os.path.exists(self.file_name):
            return
        with open(self.file_name, 'wb') as f:
            pickle.dump(self, f);

    def load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, 'rb') as f:
            new_obj = pickle.load(f)
        map_location = self.map_location
        self.__dict__ = copy(new_obj.__dict__)
        self.map_location = map_location
        return True
