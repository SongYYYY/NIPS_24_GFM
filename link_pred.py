import sys
sys.path.append("../..") 
sys.path.append('..')
import torch
import scipy.sparse as ssp
from scoring import SymmetricScoringFunction, AsymmetricScoringFunction

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch.nn as nn

from ogb.linkproppred import Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from GFM_utils import generate_link_prediction_data_, reset_gnn_weights, set_random_seed
import tqdm
import dgl
import numpy as np
import os

def train_epoch(score_func, train_pos, x, optimizer, batch_size):
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        num_nodes = x.size(0)

        h = x

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        if isinstance(score_func, AsymmetricScoringFunction):
            pos_out = score_func(h[edge[1]], h[edge[0]])
            pos_loss += -torch.log(pos_out + 1e-15).mean()
            pos_loss = pos_loss / 2

        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

def train(x, train_pos, data, score_func, device):
    mrr_list = []
    for run in range(3):
        train_seed = run
        set_random_seed(train_seed)

        score_func.reset_parameters()   
        optimizer = torch.optim.Adam(score_func.parameters(), lr=1e-3, weight_decay=0)
    
        best_mrr = 0
        count = 0
        patience = 10
        for epoch in range(1000):
            loss = train_epoch(score_func, train_pos, x, optimizer, 4096)
            
            if epoch % 5 == 0:
                mrr = evaluate_mrr(device, data, x, score_func, 'test')

                if best_mrr < mrr:
                    best_mrr = mrr
                    count = 0
                else:
                    count += 1
                    if count >= patience:
                        break

        mrr_list.append(best_mrr)

    return mrr_list

@torch.no_grad()
def compute_mrr(
    score_func, evaluator, node_emb, src, dst, neg_dst, device, batch_size=2048
):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc="Evaluate"):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device) # (N, 1, d)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device) # (N, K+1, d)

        pred = score_func(h_src.expand(h_dst.shape[0], h_dst.shape[1], h_src.shape[2]), h_dst).squeeze() # (N, K+1)
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()

@torch.no_grad()
def evaluate_mrr(device, data, x, score_func, split='test'):
    score_func.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    with torch.no_grad():
        node_emb = x
        src = data[f'{split}_pos'].t()[0].to(node_emb.device) # test_pos: [n, 2] -> src: [n]
        dst = data[f'{split}_pos'].t()[1].to(node_emb.device) # dst: [n]
        neg_dst = data[f'{split}_neg'].to(node_emb.device) # test_neg: [n, K]
        mrr = compute_mrr(
                score_func, evaluator, node_emb, src, dst, neg_dst, device
            )
    return mrr


def get_score_model(score_model, hidden_channels, score_layers, dropout):
    if score_model == 'sym':
        score_func = SymmetricScoringFunction(hidden_channels, hidden_channels, 1, score_layers, dropout)
    elif score_model == 'asym':
        score_func = AsymmetricScoringFunction(hidden_channels, hidden_channels, 1, score_layers, dropout)
    else:
        raise TypeError(f'Unrecognized Score Function! {score_model}')
    
    return score_func

@torch.no_grad()
def compute_node_emb_in_batches(model, input_context, input_tensor, edges, 
                                batch_size, agg='self', add_cls=False, device=None):
    # get trm output in batches
    model.eval()
    start = 0
    all_embeddings = []
    # compute degrees
    out_degrees = torch.bincount(edges[0], minlength=input_context.shape[0]).to(device)
    in_degrees = torch.bincount(edges[1], minlength=input_context.shape[0]).to(device)

    while start < input_tensor.shape[0]:
        input_tensor_batch = input_tensor[start:start+batch_size] # (b, l+1, d)
        input_tensor_batch = input_tensor_batch.to(device)
        input_context_batch = input_context[start:start+batch_size] # (b, l+1)
        embeddings = model.inference(input_tensor_batch, input_context_batch, 
                                     in_degrees, out_degrees, add_cls=add_cls) # (b, l+1, d)
        if add_cls:
            embeddings = embeddings[:, 1:, :]

        if agg == 'self':
            embeddings = embeddings[:, 0, :] # (b, d)
        elif agg == 'mean':
            embeddings = embeddings.mean(dim=1) # (b, d)
        elif agg == 'agg':
            center_ids = input_context_batch[:, 0] # (b,)
            node_masks = center_ids.unsqueeze(1).unsqueeze(2) == input_context_batch # (b, b, l+1)
            res = torch.zeros((embeddings.shape[0], embeddings.shape[2]), device=device)
            for i in range(res.shape[0]):
                res[i] = embeddings[node_masks[i]].mean(axis=0)
            embeddings = res
        else:
            raise NotImplementedError
        
        all_embeddings.append(embeddings)
        start += batch_size

    # Concatenate all batch similarities
    return torch.cat(all_embeddings, dim=0) # (n, d)

def get_rw_input(dgl_graph, p, q, walk_length, seed=0):
    # prepare the input tensor needed for the transformer
    set_random_seed(seed)
    dgl.seed(seed)
    start_nodes = torch.arange(dgl_graph.number_of_nodes())  # Starting nodes for walks
    walks = dgl.sampling.node2vec_random_walk(dgl_graph, start_nodes, p, q, walk_length) # (n, l+1)
    x = dgl_graph.ndata['feat'][walks] # (n, l+1, d)

    return walks, x

def get_bert_embeddings(x, adj, bert_model, agg, p, q, walk_length, seed, add_cls, device):
    # Create a DGL graph from the edges
    row, col, value = adj.coo()
    g = dgl.graph((row.cpu(), col.cpu()), num_nodes=x.shape[0])
    g.ndata['feat'] = x.cpu()
    edges = torch.stack(g.edges(), dim=0)
    input_context, input_tensor = get_rw_input(g, p, q, walk_length, seed)
    node_embs = compute_node_emb_in_batches(bert_model, input_context, input_tensor, 
                                            edges, 4096, agg, add_cls, device)

    return node_embs

def eval_all_link(link_data_all, model, agg, p, q, walk_length, add_cls, device):
    # eval_graph: {'data_name': (input_tensor, pyg_graph)}
    res_dict = {}
    for k, data in link_data_all.items():
        x = get_bert_embeddings(data['x'], data['adj'], model, agg, p, q, walk_length, 0, add_cls, device)
        score_func = get_score_model('sym', x.shape[1], 3, 0.1).to(device)
        test_mrr_list = train(x.to(device), data['train_pos'].to(device), data, score_func, device)
        res_dict[k] = [np.mean(test_mrr_list), np.std(test_mrr_list)]
    
    return res_dict



def get_link_data_all(data_names, train_ratio, val_ratio, K, seed, data_dir, link_dir):
    link_data_all = {}
    for data_name in data_names:
        pyg_data = torch.load(os.path.join(data_dir, f'{data_name}_fixed_sbert.pt'))
        data = generate_link_prediction_data_(None, data_name, train_ratio=train_ratio, valid_ratio=val_ratio, K=K, 
                                              seed=seed, save_dir=link_dir)
        data['x'] = pyg_data.x
        link_data_all[data_name] = data 

    return link_data_all

