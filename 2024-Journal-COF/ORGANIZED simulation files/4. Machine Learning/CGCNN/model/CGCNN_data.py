import os
import csv
import json
import random
import torch
import pickle
import functools
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def collate_pool(dataset_list):
    batch_atom_fea  = []
    batch_nbr_fea = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_cif_ids = []
    base_idx = 0
    for _, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
            torch.stack(batch_target, dim=0),\
            batch_cif_ids

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class  CIFData(Dataset):
    pd.options.mode.chained_assignment = None
    def __init__(self,root_dir, data_file, unit, tar, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=1129):
        self.root_dir = root_dir
        self.unit = unit
        self.tar = tar
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        id_prop_file = data_file
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [[x.strip().replace('\ufeff', '') for x in row] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir,'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return len(self.id_prop_data)
    @functools.lru_cache(maxsize=None)  
    def __getitem__(self, idx):
        cif_id, _ = self.id_prop_data[idx]
        cif_id = cif_id.replace('ï»¿', '')
        if self.tar:
            target = torch.Tensor(np.load(self.root_dir + 'npy_' + self.unit + "/" + cif_id+'.npy').reshape(15,)).float()
            if os.path.exists(self.root_dir + "pkl/" + cif_id+'.pkl'):
                with open(self.root_dir + "pkl/" + cif_id+'.pkl', 'rb') as f:
                    pkl_data = pickle.load(f)
                    atom_fea = pkl_data[0]
                    nbr_fea = pkl_data[1]
                    nbr_fea_idx = pkl_data[2]
            else:
                try:
                    crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
                    atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
                    atom_fea = torch.Tensor(atom_fea)
                    all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
                    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
                    nbr_fea_idx, nbr_fea = [], []
                    for nbr in all_nbrs:
                        if len(nbr) < self.max_num_nbr:
                            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                        else:
                            nbr_fea_idx.append(list(map(lambda x: x[2],nbr[:self.max_num_nbr])))
                            nbr_fea.append(list(map(lambda x: x[1],nbr[:self.max_num_nbr])))
                    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
                    nbr_fea = self.gdf.expand(nbr_fea)
                    atom_fea = torch.Tensor(atom_fea)
                    nbr_fea = torch.Tensor(nbr_fea)
                    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
                    with open(self.root_dir + "pkl/" + cif_id+'.pkl', 'wb') as f:
                        pickle.dump((atom_fea, nbr_fea, nbr_fea_idx), f)
                except:
                    print(cif_id)
            return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
        else:
            if os.path.exists(self.root_dir + "pkl/" + cif_id+'.pkl'):
                with open(self.root_dir + "pkl/" + cif_id+'.pkl', 'rb') as f:
                    pkl_data = pickle.load(f)
                    atom_fea = pkl_data[0]
                    nbr_fea = pkl_data[1]
                    nbr_fea_idx = pkl_data[2]
            else:
                try:
                    crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
                    atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
                    atom_fea = torch.Tensor(atom_fea)
                    all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
                    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
                    nbr_fea_idx, nbr_fea = [], []
                    for nbr in all_nbrs:
                        if len(nbr) < self.max_num_nbr:
                            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                        else:
                            nbr_fea_idx.append(list(map(lambda x: x[2],nbr[:self.max_num_nbr])))
                            nbr_fea.append(list(map(lambda x: x[1],nbr[:self.max_num_nbr])))
                    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
                    nbr_fea = self.gdf.expand(nbr_fea)
                    atom_fea = torch.Tensor(atom_fea)
                    nbr_fea = torch.Tensor(nbr_fea)
                    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
                    with open(self.root_dir + "pkl/" + cif_id+'.pkl', 'wb') as f:
                        pickle.dump((atom_fea, nbr_fea, nbr_fea_idx), f)
                except:
                    print(cif_id)
            return (atom_fea, nbr_fea, nbr_fea_idx), cif_id

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=32,random_seed = 2, val_ratio=0.2,
                              test_ratio=0.2, num_workers=8, pin_memory=False):
    total_size = len(dataset)
    train_ratio = 1 - val_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", random_seed)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(train_size, valid_size, test_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size):-test_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=train_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=test_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
    
