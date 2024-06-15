import torch
import numpy as np
import dgl
import copy
from dgl import GCNNorm
import random

def set_random_seed(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python
    random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


class Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, g):
        self.len = g.number_of_edges()

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        return self.len


class Bert_Collator(object):
    def __init__(self, device, 
                p, q, walk_length,
                mask_rate,
                batch_size,
                undirected=False,
                return_edges=False,
                g_aug_rate=0,
                negative_method='default',
                add_extra_negative=False,
                ):

        self.device = device
        self.mask_rate = mask_rate
        self.p = p
        self.q = q 
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.undirected = undirected
        self.return_edges = return_edges
        self.g_aug_rate = g_aug_rate
        self.negative_method = negative_method
        self.add_extra_negative = add_extra_negative
        

    def __call__(self, gs):
        g = gs[0]
        node_features = g.ndata['feat']
        edges = g.edges()
        in_degrees = torch.bincount(edges[1], minlength=node_features.shape[0])
        if self.undirected:
            g = dgl.add_reverse_edges(g)

        if self.g_aug_rate > 0:
            n_aug_edges = int(g.num_edges() * self.g_aug_rate)
            edge_ids = torch.randperm(g.num_edges())[:n_aug_edges]
            src, dst = g.edges()
            g = dgl.add_edges(g, dst[edge_ids], src[edge_ids])
        
        # print('#Nodes: {}, #Edges: {}'.format(g.num_nodes(), g.num_edges()))

        start_nodes = torch.arange(g.num_nodes())  # Starting nodes for walks

        walks = dgl.sampling.node2vec_random_walk(g, start_nodes, self.p, self.q, self.walk_length)

        if self.negative_method == 'default':
            # leave unchanged
            rand_id = -1 * torch.ones_like(walks)
        elif self.negative_method == 'random':
            valid_mask = (walks != -1)
            rand_id = torch.randint(0, g.num_nodes(), (walks.shape[0], 1)).expand_as(walks)
            walks = torch.where(valid_mask, walks, rand_id)
        else:
            raise NotImplementedError
        
        if self.add_extra_negative:
            neg_mask = self.create_random_mask(walks.shape[0], walks.shape[1])
            walks = torch.where(neg_mask, rand_id, walks)

        if self.batch_size != 0:
            if walks.shape[0] > self.batch_size:
                idxs = torch.randperm(walks.shape[0])[:self.batch_size]
                walks = walks[idxs]
            else:
                gap = self.batch_size - walks.shape[0]
                padding_idxs = torch.randint(0, walks.shape[0], size=(gap,))
                padding = walks[padding_idxs]
                walks = torch.concat([walks, padding], dim=0)

        target_mask = torch.rand(walks.shape) < self.mask_rate
        
        if not self.return_edges:
            return node_features, walks, target_mask, in_degrees
        else:
            edges = torch.stack(g.edges(), dim=0)
            return node_features, walks, target_mask, in_degrees, edges
        
    def create_random_mask(self, rows, cols):
        col_indices = torch.arange(cols).unsqueeze(0)  # Shape: [1, cols]
        start_indices = torch.randint(1, cols, (rows, 1))  # Shape: [rows, 1]
        mask = col_indices >= start_indices 

        return mask


