import sys
sys.path.append("..") 

import torch
import argparse
import scipy.sparse as ssp
from projectors import SymmetricScoringFunction, AsymmetricScoringFunction, ContrastiveScoringFunction
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch.nn as nn
import dgl 
import dgl.nn as dglnn
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from GFM_utils import generate_link_prediction_data_, reset_gnn_weights, set_random_seed
import tqdm
import os
import fitlog
from dgl import GCNNorm
from copy import deepcopy
from utils import construct_features_sparse, dgl_to_torch_sparse, init_process_group
from mssl_model import PretrainModule
from model import TransformerModel
from torch_geometric.utils import to_undirected, remove_self_loops
import torch.multiprocessing as mp
os.environ["DGLBACKEND"] = "pytorch"
import torch.distributed as dist
    


def train(model, score_func, train_pos, x, optimizer, batch_size, hops):
    model.train()
    score_func.train()
    transform = GCNNorm()
    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0) 

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
    
        train_edge_mask = train_pos[mask].transpose(1,0)

        # train_edge_mask = to_undirected(train_edge_mask)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        row, col, value = adj.coo()  # row: source nodes, col: destination nodes

        # Create a DGL graph from the edges
        g = dgl.graph((row, col), num_nodes=x.shape[0]).to(train_pos.device)

        g = transform(g)
        adj = dgl_to_torch_sparse(g)
        input_tensor = construct_features_sparse(adj, x, hops)

        h = model(input_tensor)

        edge = train_pos[perm].t()

        pos_out = F.sigmoid(score_func(h[edge[0]], h[edge[1]]))
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        if isinstance(score_func, AsymmetricScoringFunction):
            pos_out = F.sigmoid(score_func(h[edge[1]], h[edge[0]]))
            pos_loss += -torch.log(pos_out + 1e-15).mean()
            pos_loss = pos_loss / 2

        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = F.sigmoid(score_func(h[edge[0]], h[edge[1]]))
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

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


def evaluate_mrr(device, data, model, score_func, hops, split='test'):
    transform = GCNNorm()
    model.eval()
    score_func.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    adj = data['adj']
    row, col, value = adj.coo()  # row: source nodes, col: destination nodes
    # Create a DGL graph from the edges
    g = dgl.graph((row, col), num_nodes=data['x'].shape[0]).to(device)
    g = transform(g)
    adj = dgl_to_torch_sparse(g)
    input_tensor = construct_features_sparse(adj, data['x'].to(device), hops)
    with torch.no_grad():
        node_emb = model(input_tensor)
        src = data[f'{split}_pos'].t()[0].to(node_emb.device) # test_pos: [n, 2] -> src: [n]
        dst = data[f'{split}_pos'].t()[1].to(node_emb.device) # dst: [n]
        neg_dst = data[f'{split}_neg'].to(node_emb.device) # test_neg: [n, K]
        mrr = compute_mrr(
                score_func, evaluator, node_emb, src, dst, neg_dst, device
            )
    return mrr

def set_model_state(ckpt_dir, ckpt_name, epoch, pretrain_model):
    if ckpt_name != 'none':
        # from collections import OrderedDict
        ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}-{epoch}.ckpt')
        # checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v

        # pretrain_model.load_state_dict(new_state_dict, strict=True)

        pretrain_model.load_state_dict(torch.load(ckpt_path, map_location='cpu').module.state_dict(), strict=True)
        # backbone = pretrain_model.model
        print(f'backbone loaded from {ckpt_path}.')
    else:
        # backbone = pretrain_model.model
        reset_gnn_weights(pretrain_model.model)
        print('backbone init.')

    return 


def add_hyper(params):
    tmp_dict={}
    for k,v in params.items():
        if isinstance(v,tuple) or isinstance(v,list):
            tmp_dict[k]=v
    fitlog.add_hyper(params)
    for k,v in tmp_dict.items():
        params[k]=v

def init_fitlog(param_grid, log_dir='logs'):
    fitlog.commit(__file__)
    fitlog.set_log_dir(log_dir)
    add_hyper(param_grid)

def train_and_eval(rank, param_grid):
    init_process_group(1, rank, 15223)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device(f"cuda:{rank}")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    print(device)

    log_dir = param_grid['log_dir']
    init_fitlog(param_grid, log_dir)
    print('Fitlog init.')

    seed = param_grid['seed']
    set_random_seed(seed)

    # data 
    data_name = param_grid['data_name']
    train_ratio = param_grid['train_ratio']
    val_ratio = param_grid['val_ratio']
    data_seed = param_grid['data_seed']
    # model
    hops = param_grid['hops']
    projector_type = param_grid['projector_type']
    n_layers = param_grid['n_layers']
    n_heads = param_grid['n_heads']
    projector_layers = param_grid['projector_layers']
    hidden_dim = param_grid['hidden_dim']
    projector_dim = param_grid['projector_dim']
    dropout = param_grid['dropout']
    attention_dropout = param_grid['attention_dropout']
    # train
    freeze = param_grid['freeze']
    ckpt_dir = param_grid['ckpt_dir']
    ckpt_name = param_grid['ckpt_name']
    pretrain_epoch = param_grid['pretrain_epoch']
    batch_size = param_grid['batch_size']
    lr = param_grid['lr']
    epochs = param_grid['epochs']
    eval_steps = param_grid['eval_steps']
    runs = param_grid['runs']
    patience = param_grid['patience']
    l2 = param_grid['l2']
    # evaluation
    K = param_grid['K']

    set_random_seed(seed)

    downstream_root = '/local/scratch/ysong31/HeaRT/GFM/LLM_data'
    data_ori = torch.load(os.path.join(downstream_root, f'{data_name}_fixed_sbert.pt'))

    data = generate_link_prediction_data_(data_ori, data_name, train_ratio=train_ratio, valid_ratio=val_ratio, K=K, seed=data_seed,
                                          save=True, save_dir='../link_data')
    data['x'] = data_ori.x
   
    x = data['x']
    x = x.to(device)
    train_pos = data['train_pos'].to(x.device)

    # model configuration
    trm_model = TransformerModel(hops=hops, 
                                input_dim=384, 
                                n_layers=n_layers,
                                num_heads=n_heads,
                                hidden_dim=hidden_dim,
                                dropout_rate=dropout,
                                attention_dropout_rate=attention_dropout).to(device)
    
    pretrain_model = PretrainModule(trm_model, hidden_dim, projector_dim, hops).to(device)
    
    if projector_type == 'Sym':
        projector = SymmetricScoringFunction(hidden_dim*2, hidden_dim, 1, projector_layers, 0.2)
    elif projector_type == 'Asym':
        projector = AsymmetricScoringFunction(hidden_dim*2, hidden_dim, 1, projector_layers, 0.2)
    elif projector_type == 'Con':
        projector = ContrastiveScoringFunction(hidden_dim*2, hidden_dim, 128, projector_layers, 0.2)
    else:
        raise TypeError(f'Unrecognized Scoring Function: {projector_type}!')

    projector.to(device)

    mrr_list = []
    epoch_list = []
    for run in range(runs):

        print('#################################          ', run, '          #################################')
        
        if runs == 1:
            train_seed = seed
        else:
            train_seed = run
        print('train seed: ', train_seed)

        set_random_seed(train_seed)

        set_model_state(ckpt_dir, ckpt_name, pretrain_epoch, pretrain_model)
        reset_gnn_weights(projector)
        print('Projector init.')

        if freeze:
            for param in trm_model.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(projector.parameters(), lr=lr, weight_decay=l2)
        else:
            optimizer = torch.optim.Adam(
                    list(trm_model.parameters()) + list(projector.parameters()), lr=lr, weight_decay=l2)

        best_valid = 0
        best_epoch = -1
        count = 0
        for epoch in range(1, 1 + epochs):
            loss = train(trm_model, projector, train_pos, x, optimizer, batch_size, hops)
            
            if epoch % eval_steps == 0:
                mrr = evaluate_mrr(device, data, trm_model, projector, hops, 'valid')
                print('Valid MRR: {:.4f}'.format(mrr))

                if best_valid < mrr:
                    best_valid = mrr
                    best_epoch = epoch
                    count = 0
                    backbone_weights = deepcopy(trm_model.state_dict())
                    projector_weights = deepcopy(projector.state_dict())
                else:
                    count += 1
                    if count >= patience:
                        break 

        print('RUN: {}, Training Stop! Best MRR: {:.4f} at Epoch {}'.format(run, best_valid, best_epoch))
        trm_model.load_state_dict(backbone_weights)
        projector.load_state_dict(projector_weights)
        print('Ckpt loaded.')
        test_mrr = evaluate_mrr(device, data, trm_model, projector, hops, 'test')
        mrr_list.append(test_mrr)
        print('RUN: {}, TEST MRR: {:.4f}'.format(run, test_mrr))
        epoch_list.append(best_epoch)

    print('TEST MRR LIST: {}'.format(mrr_list))
    print('MEAN TEST MRR: {:.4f}, STD: {:.4f}, Epoch: {}'.format(np.mean(mrr_list), np.std(mrr_list), np.mean(epoch_list)))
    fitlog.add_best_metric({'MEAN': np.mean(mrr_list), 'STD': np.std(mrr_list), 'EPOCH': np.mean(epoch_list)}, name='TEST-MRR')


import time
def train_and_record(pId, param_grid):
    try:
        # train_and_eval(param_grid)
        mp.spawn(train_and_eval, args=(param_grid, ), nprocs=1)
    except Exception as e:
        print('error occured in {}.'.format(param_grid))
        print(e)
        fitlog.finish()
        # record errors
        with open('error_log.txt', 'a') as f:
            f.writelines([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), str(param_grid), str(e)])
            f.write('\n')
    # train_and_eval(param_grid)
    return



if __name__ == '__main__':
    param_grid = {
    # data
    'data_name': 'cora',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'data_seed': 0,
    # model 
    'hops': 3,
    'projector_type': 'Sym',
    'n_layers': 2,
    'n_heads': 8,
    'projector_layers': 3,
    'hidden_dim': 384,
    'projector_dim': 256,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    # train
    'freeze': False,
    'ckpt_dir': '/local/scratch/ysong31/HeaRT/GFM/NAGPhormer/ckpt-concat-pro2d',
    'ckpt_name': "['p_link', 'p_recon']",
    'pretrain_epoch': 99,
    # 'ckpt_name': "none",
    # 'ckpt_name': "['p_recon']",
    # 'ckpt_name': "['p_link']",
    'batch_size': 4096,
    'lr': 0.001,
    'l2': 0,
    'epochs': 1000,
    'patience': 20,
    'runs': 3,
    'eval_steps': 5,
    # evaluation
    'K': 100,
    # others
    'seed': 123,
    'log_dir': 'logs',
    }
    

    # train_and_eval(param_grid)
    mp.spawn(train_and_eval, args=(param_grid, ), nprocs=1)
