import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def construct_loader(data_path, nb_samples, batch_size, shuffle=False):
    dataset = IceCube_Dataset(data_path, nb_samples)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=True,
                        collate_fn=collate_icecube,
                        pin_memory=True,
                        num_workers=0)
    return loader

def collate_icecube(samples):
    X = [s[0] for s in samples]
    y = [s[1] for s in samples]
    w = [s[2] for s in samples]
    evt_ids   = [s[3] for s in samples]
    evt_names = [s[4] for s in samples]
    energy    = [s[5] for s in samples]
    reco      = [s[6] for s in samples]

    X, adj_mask, batch_nb_nodes = pad_batch(X)
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    w = torch.FloatTensor(w)
    adj_mask = torch.FloatTensor(adj_mask)
    batch_nb_nodes = torch.FloatTensor(batch_nb_nodes)
    return X, y, w, adj_mask, batch_nb_nodes, evt_ids, evt_names, energy, reco

def pad_batch(X):
    nb_samples = len(X)
    nb_features = X[0].shape[1]
    batch_nb_nodes = [s.shape[0] for s in X]
    largest_size = max(batch_nb_nodes)

    adj_mask = np.zeros(shape=(nb_samples, largest_size, largest_size))
    # Append zero nodes to features with fewer points in point cloud
    #   than largest_size.
    for i in range(nb_samples):
        zeros = np.zeros((largest_size-X[i].shape[0], nb_features))
        X[i] = np.concatenate((X[i], zeros), axis=0)
        adj_mask[i, :batch_nb_nodes[i], :batch_nb_nodes[i]] = 1

    # Put all samples into numpy array.
    X = np.stack(X, axis=0)
    return X, adj_mask, batch_nb_nodes

class IceCube_Dataset(Dataset):
    def __init__(self, data_path, nb_samples):
        with open(data_path, 'rb') as f:
            X, y, weights, event_id, filenames, energy, reco = pickle.load(f)
        self.X = X[:nb_samples]
        self.y = y[:nb_samples]
        self.w = weights[:nb_samples]
        self.e = event_id[:nb_samples]
        self.f = filenames[:nb_samples]
        self.E = energy[:nb_samples]
        self.r = reco[:nb_samples]

    def __getitem__(self, index):
        X_i = self.X[index] # leave out nstring
        y_i = self.y[index]
        w_i = self.w[index]
        e_i = self.e[index]
        f_i = self.f[index]
        E_i = self.E[index]
        r_i = self.r[index]
        return X_i, y_i, w_i, e_i, f_i, E_i, r_i

    def __len__(self):
        return len(self.w)

