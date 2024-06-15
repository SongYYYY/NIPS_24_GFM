import utils
import dgl
import torch
import scipy.sparse as sp
import random
import numpy as np
from torch.utils.data import Dataset

class LazyGraphDataset(Dataset):
    def __init__(self, graph_list, features_path):
        """
        Initialize the dataset.

        :param graph_list: A list of DGL Graph objects.
        :param features_path: Path to the features tensor saved on disk.
        """
        self.graph_list = graph_list
        self.features = None
        self.features_path = features_path

    def __len__(self):
        """
        Return the length of the dataset (number of graphs).
        """
        return len(self.graph_list)

    def __getitem__(self, idx):
        """
        Return a single graph and its corresponding features.

        :param idx: Index of the graph in the dataset.
        """
        if self.features is None:
            # self.features = np.memmap(self.features_path, dtype='float32', mode='r')
            self.features = np.load(self.features_path, mmap_mode='r')

        graph = self.graph_list[idx]

        # Check if 'dgl.NID' feature exists
        if dgl.NID in graph.ndata:
            nids = graph.ndata[dgl.NID]
            # Load the corresponding features from the features tensor
            graph.ndata['feat'] = torch.tensor(self.features[nids.numpy()], dtype=torch.float32)
            del graph.ndata[dgl.NID]
        else:
            raise KeyError(f"{dgl.NID} feature not found in graph")

        return graph

class GraphDataset(Dataset):
    def __init__(self, graph_list, features):
        """
        Initialize the dataset.

        :param graph_list: A list of DGL Graph objects.
        :param features: The shared-memory features tensor.
        """
        self.graph_list = graph_list
        # Load the features tensor using memory mapping
        self.features = features

    def __len__(self):
        """
        Return the length of the dataset (number of graphs).
        """
        return len(self.graph_list)

    def __getitem__(self, idx):
        """
        Return a single graph and its corresponding features.

        :param idx: Index of the graph in the dataset.
        """
        graph = self.graph_list[idx]

        # Check if 'dgl.NID' feature exists
        if dgl.NID in graph.ndata:
            nids = graph.ndata[dgl.NID]
            # Load the corresponding features from the features tensor
            graph.ndata['feat'] = self.features[nids]
        else:
            raise KeyError("dgl.NID feature not found in graph")

        return graph
    


class RWDataset(Dataset):
    def __init__(self, mask):
        self.filtered_indices = torch.nonzero(mask, as_tuple=False).squeeze()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Return the actual node index corresponding to the current dataset index
        return self.filtered_indices[idx].item()

