import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
import random



def dgl_to_torch_sparse(g):
    # Get the adjacency matrix in COO format
    src, dst = g.edges()

    # Create the edge index in PyTorch COO format
    edge_index = torch.stack([src, dst], dim=0)

    # If the graph has edge weights, use them as values
    if 'w' in g.edata:
        edge_weights = g.edata['w']
    else:
        # If no edge weights, use a tensor of ones
        edge_weights = torch.ones(edge_index.shape[1])

    # Create the sparse adjacency matrix
    num_nodes = g.num_nodes()
    adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weights, (num_nodes, num_nodes))

    return adj_matrix

def construct_features_sparse(adj_matrix, X, K):
    # Initialize X_new with self-features
    X_new = [X]

    # Current features for propagation
    current_features = X

    # Iteratively propagate features
    for _ in range(K):
        # Sparse matrix multiplication for feature propagation
        current_features = torch.sparse.mm(adj_matrix, current_features)
        X_new.append(current_features)

    # Concatenate along a new dimension to form [N, K+1, d]
    X_new = torch.stack(X_new, dim=1)

    return X_new


def dgl_to_scipy_sparse(g):
    # Get the edges of the graph
    src, dst = g.edges()

    # Optionally, handle edge weights if your graph has them
    if 'edge_weight' in g.edata:
        edge_weight = g.edata['edge_weight'].numpy()
    else:
        edge_weight = torch.ones(src.shape[0]).numpy()  # Use 1s if no weights

    # Number of nodes
    num_nodes = g.num_nodes()

    # Create a SciPy sparse matrix in COO format
    adj_matrix = sp.coo_matrix((edge_weight, (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))

    # Convert to CSR format
    adj_matrix_csr = adj_matrix.tocsr()

    return adj_matrix_csr


def concat_dgl_graphs(graph_list):
    # Check if the graph list is empty
    if not graph_list:
        raise ValueError("The graph list is empty")

    # Concatenate edge connections and adjust edge indices
    src_list = []
    dst_list = []
    offset = 0
    for graph in graph_list:
        src, dst = graph.edges()
        src_list.append(src + offset)
        dst_list.append(dst + offset)
        offset += graph.number_of_nodes()

    src_cat = torch.cat(src_list, dim=0)
    dst_cat = torch.cat(dst_list, dim=0)

    # Create the concatenated graph
    concatenated_graph = dgl.graph((src_cat, dst_cat), num_nodes=offset)

    return concatenated_graph


import torch.distributed as dist
def print_mean_loss(rank, loss, world_size):
    """
    Gather losses from all processes, compute the mean, and print on rank 0.
    
    Args:
    - rank (int): The rank of the current process.
    - loss (torch.Tensor): The loss of the current process.
    - world_size (int): Total number of processes.
    """
    # Ensure that loss is a tensor
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, device='cuda')

    # Gather all losses to rank 0
    all_losses = [torch.tensor(0.0, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_losses, loss)

    if rank == 0:
        # Only rank 0 computes the mean and prints it
        mean_loss = sum(all_losses) / world_size
        print(f"Mean Loss: {mean_loss.item()}")


def init_process_group(world_size, rank, port=12345):
    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )

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

import time 
def estimate_remaining_time(start_time, current_batch, total_batches, k):
    """
    Estimates and prints the remaining training time every K batches.

    :param start_time: The time when the epoch started.
    :param current_batch: The current batch number.
    :param total_batches: The total number of batches in the epoch.
    :param k: The function will print the estimated time every K batches.
    """

    if current_batch % k == 0 and current_batch > 0:
        elapsed_time = time.time() - start_time
        batches_processed = current_batch
        avg_time_per_batch = elapsed_time / batches_processed
        remaining_batches = total_batches - current_batch
        estimated_time = avg_time_per_batch * remaining_batches

        # Convert estimated time to minutes and seconds for better readability
        estimated_minutes = int(estimated_time // 60)
        estimated_seconds = int(estimated_time % 60)

        print(f"Estimated Time Remaining: {estimated_minutes}m {estimated_seconds}s")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reset_gnn_weights(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()