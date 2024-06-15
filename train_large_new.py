import utils 
import dgl
import torch
from dgl import GCNNorm
import dgl.function as fn
import torch.nn.functional as F
from bert import GraphBert, MLPDecoder, GraphBertEmbeddings, BertEncoder, IdentityMapping
from lr import PolynomialDecayLR
import os.path
import argparse
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import init_process_group, print_mean_loss, estimate_remaining_time, worker_init_fn
from torch.utils.data import DataLoader, DistributedSampler
from mssl_new import Bert_Collator
from data import LazyGraphDataset
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
import numpy as np
import time 
import datetime
import gc
import random
import transformers
transformers.logging.set_verbosity_info()
from fitlog_utils import init_fitlog, create_folder
import fitlog
from link_pred import get_link_data_all, eval_all_link

@torch.no_grad()
def compute_node_emb_in_batches(model, input_context, input_tensor, edges, 
                                batch_size, agg='self', add_cls=False, device=None):
    # prepare input data
    model.eval()
    start = 0
    all_embeddings = []
    # compute degrees
    in_degrees = torch.bincount(edges[1], minlength=input_context.shape[0]).to(device)
    out_degrees = torch.zeros_like(in_degrees)
 
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

    return torch.cat(all_embeddings, dim=0) # (n, d)

def eval_downstream(input_context, input_tensor, pyg_graph, model, agg, add_self, add_cls, device):
    labels = pyg_graph.y 
    edges = pyg_graph.edge_index
    node_embs = compute_node_emb_in_batches(model, input_context, input_tensor, edges, 2048, agg, add_cls, device)
    if add_self:
        node_embs = torch.cat([pyg_graph.x.to(device), node_embs], dim=1)
    # node_embs = pyg_graph.x.to(device)
    n_classes = labels.max().item() + 1

    test_acc_list = []
    test_loss_list = []
    for split in range(len(pyg_graph.train_masks)):
        clf = torch.nn.Linear(node_embs.shape[1], n_classes).to(device)
        test_acc, test_loss = train_clf(clf, node_embs, labels, pyg_graph.train_masks[split], pyg_graph.test_masks[split], device)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    return test_acc_list, test_loss_list

def train_clf(clf, node_embs, labels, train_mask, test_mask, device):
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0)
    best_acc = 0
    best_loss = 0
    for e in range(100):
        clf.train()
        optimizer.zero_grad()
        out = clf(node_embs[train_mask].to(device))
        loss = F.cross_entropy(out, labels[train_mask].to(device))
        loss.backward()
        optimizer.step()
        test_acc, test_loss = evaluate_clf(clf, node_embs, labels, test_mask, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = test_loss
        
    return best_acc, best_loss


@torch.no_grad()
def evaluate_clf(clf, node_embs, labels, mask, device):
    clf.eval()
    out = clf(node_embs[mask].to(device))
    pred = out.argmax(dim=1)
    correct = pred.eq(labels[mask].to(device)).sum().item()
    acc = correct / mask.sum().item()
    loss = F.cross_entropy(out, labels[mask].to(device)).item()
    return acc, loss

def eval_all(eval_graphs, model, agg, add_self, add_cls, device):
    # eval_graph: {'data_name': (input_tensor, pyg_graph)}
    res_dict = {}
    for k in eval_graphs.keys():
        test_acc_list, test_loss_list = eval_downstream(eval_graphs[k][0], eval_graphs[k][1], eval_graphs[k][2],
                                                        model, agg, add_self, add_cls, device) 
        res_dict[k] = [np.mean(test_acc_list), np.mean(test_loss_list)]
    
    return res_dict


def train(dataloader, sampler, model, optimizer, scheduler, scaler, tasks, epochs, rank, device, 
          ckpt_path, eval_graphs, link_data_all, agg, p, q, walk_length, add_self, add_cls, alpha):
    if dist.get_rank() == 0:
        pbar = tqdm(total=epochs)
    else:
        pbar = None
    num_batches = len(dataloader)
    init_time = time.time()

    if rank == 0:
        # eval node clf
        res_dict = eval_all(eval_graphs, model.module, agg, add_self, add_cls, device)
        print("----------------Epoch {:05d}----------------".format(-1))
        for k in res_dict.keys():
            print('DATA_NAME: {} | TEST-ACC: {:.5f} | TEST-LOSS: {:.5f}'.format(k, res_dict[k][0], res_dict[k][1]))
            fitlog.add_metric({'ACC': res_dict[k][0], 'LOSS': res_dict[k][1]}, name=k, step=0)
        eval_dict = res_dict.copy()

    if rank == 0:
        # eval link pred
        res_dict = eval_all_link(link_data_all, model.module, agg, p, q, walk_length, add_cls, device)
        print("----------------Epoch {:05d}----------------".format(-1))
        for k in res_dict.keys():
            print('DATA_NAME: {} | TEST-MRR-MEAN: {:.5f} | TEST-MRR-STD: {:.5f}'.format(k, res_dict[k][0], res_dict[k][1]))
            fitlog.add_metric({'MRR-MEAN': res_dict[k][0], 'MRR-STD': res_dict[k][1]}, name=k, step=0)
        link_dict = res_dict.copy()


    for epoch in range(epochs):
        epoch_loss_mean = 0
        epoch_mlm_loss_mean = 0
        epoch_nsp_loss_mean = 0
        loss_count = 0
        lr_mean = 0

        model.train()
        sampler.set_epoch(epoch)
        step = 0
        start_time = time.time()
        for sample in dataloader:
            optimizer.zero_grad()

            with autocast():
                loss_mlm, loss_nsp = model(sample, add_cls=add_cls)  
                train_loss = loss_mlm
            
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            lr_mean += optimizer.param_groups[0]['lr']

            loss_mlm = loss_mlm.item() ** (1/alpha)

            epoch_loss_mean += train_loss.item()
            epoch_mlm_loss_mean += loss_mlm
            epoch_nsp_loss_mean += loss_nsp.item()
            loss_count += 1
            
            if rank == 0:
                print('{}-{}, TRAINING LOSS: {:.4f} | MLM LOSS: {:.4f} | NSP LOSS: {:.4f}'.format(epoch, 
                                                                                                    step, 
                                                                                                    train_loss.item(),
                                                                                                    loss_mlm,
                                                                                                    loss_nsp.item()))
                
                estimate_remaining_time(start_time, step, num_batches, 100)
            
            step += 1
        
        if rank == 0 and epoch % 1 == 0:
            torch.save(model.module.state_dict(), os.path.join(ckpt_path, f'{tasks}-{epoch}.ckpt'))

        if dist.get_rank() == 0:
            pbar.update(1)

        if rank == 0:
            epoch_time = time.time()
            running_time = epoch_time - init_time
            formatted_time = str(datetime.timedelta(seconds=running_time))
            print('TOTAL RUNNING TIME: EPOCH-{}: {}.'.format(epoch, formatted_time))

        if rank == 0:
            print('EPOCH {:05d} | MEAN LR: {:.8f} '.format(
                epoch, lr_mean / loss_count
            ))

        if rank == 0:
            print('EPOCH {:05d} | TRAIN LOSS MEAN: {:.5f} | MLM LOSS MEAN: {:.5f} | NSP LOSS MEAN: {:.5f}'.format(
                epoch, epoch_loss_mean/loss_count, epoch_mlm_loss_mean/loss_count, epoch_nsp_loss_mean/loss_count
            ))
            fitlog.add_loss({'LOSS': epoch_loss_mean/loss_count,
                                'MLM_LOSS': epoch_mlm_loss_mean/loss_count,
                                'NSP_LOSS': epoch_nsp_loss_mean/loss_count}, name='TRAIN', step=epoch+1)

        if rank == 0:
            # eval node clf
            res_dict = eval_all(eval_graphs, model.module, agg, add_self, add_cls, device)
            print("----------------Epoch {:05d}----------------".format(epoch))
            for k in res_dict.keys():
                print('DATA_NAME: {} | TEST-ACC: {:.5f} | TEST-LOSS: {:.5f}'.format(k, res_dict[k][0], res_dict[k][1]))
                fitlog.add_metric({'ACC': res_dict[k][0], 'LOSS': res_dict[k][1]}, name=k, step=epoch+1)
            for k in eval_dict.keys():
                if eval_dict[k][0] < res_dict[k][0]:
                    eval_dict[k][0] = res_dict[k][0]
                    eval_dict[k][1] = res_dict[k][1]

        if rank == 0:
            # eval link pred
            res_dict = eval_all_link(link_data_all, model.module, agg, p, q, walk_length, add_cls, device)
            print("----------------Epoch {:05d}----------------".format(epoch))
            for k in res_dict.keys():
                print('DATA_NAME: {} | TEST-MRR-MEAN: {:.5f} | TEST-MRR-STD: {:.5f}'.format(k, res_dict[k][0], res_dict[k][1]))
                fitlog.add_metric({'MRR-MEAN': res_dict[k][0], 'MRR-STD': res_dict[k][1]}, name=k, step=epoch+1)
            for k in link_dict.keys():
                if link_dict[k][0] < res_dict[k][0]:
                    link_dict[k][0] = res_dict[k][0]
                    link_dict[k][1] = res_dict[k][1]
        

    if dist.get_rank() == 0:
        pbar.close()

    if rank == 0:
        eval_acc_mean = 0
        for k in eval_dict.keys():
            fitlog.add_best_metric({'ACC': eval_dict[k][0]}, name=f'EVAL-{k}')
            eval_acc_mean += eval_dict[k][0]
        fitlog.add_best_metric({'ACC': eval_acc_mean/len(eval_dict)}, name='EVAL-MEAN')

    if rank == 0:
        eval_mrr_mean = 0
        for k in link_dict.keys():
            fitlog.add_best_metric({'MRR': link_dict[k][0]}, name=f'LINK-{k}')
            eval_mrr_mean += link_dict[k][0]
        fitlog.add_best_metric({'MRR': eval_mrr_mean/len(link_dict)}, name='LINK-MEAN')

def load_subgraphs(data_dir, graph_name):
    n_node_list = []
    n_edge_list = []
    subgraphs = dgl.load_graphs(os.path.join(data_dir, f'{graph_name}_subgraphs.dgl'))[0]
    for i in range(len(subgraphs)):
        sg = subgraphs[i]
        n_node_list.append(sg.num_nodes())
        n_edge_list.append(sg.num_edges())

    return subgraphs, n_node_list, n_edge_list


def get_input(dgl_graph, k, l, sgc_input_rate=0, seed=0):
    # prepare the input tensor needed for the transformer
    torch.manual_seed(seed)
    emb_list = []
    context_list = []
    for i in range(dgl_graph.num_nodes()):
        subgraph, _ = dgl.khop_in_subgraph(dgl_graph, i, k=k)
        node_idx = subgraph.ndata[dgl.NID]
 
        indices = torch.randint(0, len(node_idx), (l,))
        sub_nodes = node_idx[indices]
        new_value = torch.tensor([i])  # Ensure this is a tensor, not just a scalar
        
        # Concatenate the new value tensor with the original tensor
        sub_nodes = torch.cat((new_value, sub_nodes), dim=0)
        # print(sub_nodes)

        context_list.append(sub_nodes) # (l+1)
        emb_list.append(dgl_graph.ndata['feat'][sub_nodes]) # (l+1, d)


    res_context = torch.stack(context_list, dim=0) # (n, l+1)
    x = torch.stack(emb_list, dim=0) # (n, l+1, d)
    if sgc_input_rate > 0:
        z = get_sgc_feat(dgl_graph.ndata['feat'], dgl_graph)[res_context] # (n, l+1, d)
        sgc_input_mask = torch.rand(res_context.shape) < sgc_input_rate
        res_emb = torch.where(sgc_input_mask.unsqueeze(-1).expand_as(z), z, x)
    else:
        res_emb = x

    return res_context, res_emb

def get_rw_input(dgl_graph, p, q, walk_length, sgc_input_rate=0, seed=0):
    # prepare the input tensor needed for the transformer
    utils.set_random_seed(seed)
    dgl.seed(seed)
    start_nodes = torch.arange(dgl_graph.number_of_nodes())  # Starting nodes for walks
    walks = dgl.sampling.node2vec_random_walk(dgl_graph, start_nodes, p, q, walk_length) # (n, l+1)
    x = dgl_graph.ndata['feat'][walks] # (n, l+1, d)
    if sgc_input_rate > 0:
        z = get_sgc_feat(dgl_graph.ndata['feat'], dgl_graph)[walks] # (n, l+1, d)
        sgc_input_mask = torch.rand(walks.shape) < sgc_input_rate
        res_emb = torch.where(sgc_input_mask.unsqueeze(-1).expand_as(z), z, x)
    else:
        res_emb = x

    return walks, res_emb

def get_sgc_feat(x, graph):
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    transform = GCNNorm()
    graph = transform(graph)
    adj = utils.dgl_to_torch_sparse(graph)
    z = torch.sparse.mm(adj, x)

    return z

def main(rank, world_size, seed, args):
    if args.port == 12345:
        args.port = random.randint(10000, 65535)
    init_process_group(world_size, rank, args.port)
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    args_dict = vars(args)
    init_fitlog(args_dict, args.log_dir)
    create_folder(args.ckpt_path)
    utils.set_random_seed(seed)

    data_dir = './Partition/subgraphs_original'
    graph_name = 'papers100M'
    feature_path = './Feature/sbert_embeddings_con_split.npy'
    graph_list, n_node_list, n_edge_list = load_subgraphs(data_dir, graph_name)

    if args.data_size > 0:
        ids = list(range(len(graph_list)))
        random.shuffle(ids)
        ids = ids[:args.data_size]
        graph_list = [graph_list[i] for i in ids]
        print('DATA IDS: {}'.format(ids))


    collator = Bert_Collator(device, 
                    args.p, args.q, 
                    args.walk_length,
                    args.mask_rate,
                    args.seq_batch_size,
                    undirected=args.undirected,
                    return_edges=False,
                    g_aug_rate=args.g_aug_rate,
                    negative_method=args.negative_method,
                    add_extra_negative=args.add_extra_negative,
                    )

    dataset = LazyGraphDataset(graph_list, feature_path)
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=sampler,
                    collate_fn=collator, worker_init_fn=worker_init_fn, num_workers=4, pin_memory=True)
    

    
    embeddings = GraphBertEmbeddings(384, args.hidden_dim, args.max_length, 
                                     args.max_degree, args.max_degree, args.merge_degree,
                                     dropout=args.emb_dropout, layer_norm=args.emb_layer_norm, 
                                     use_degree_emb=args.use_degree_emb,
                                     bin_degree=args.bin_degree,
                                     bin_base=args.bin_base)


    encoder = BertEncoder(args.hf_ckpt_name, args.n_layers, use_pretrained=args.hf_use_pretrained)
   
    decoder = MLPDecoder(args.mlp_dims)

    model = GraphBert(embeddings=embeddings,
                      encoder=encoder,
                      decoder=decoder,
                      walk_length=args.walk_length,
                      alpha=args.alpha,
                      p_random=args.p_random,
                      p_unchanged=args.p_unchanged,
                      pretext=args.pretext,
                      mask_center=args.mask_center,
                      init=args.init_method,
                      device=device).to(device)

    if device.type == "cpu":
        model = DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device, find_unused_parameters=True
        )

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=args.warmup_updates,
                tot_updates=args.tot_updates,
                lr=args.peak_lr,
                end_lr=args.end_lr,
                power=1.0,
            )
    scaler = GradScaler()

    # downstream data
    downstream_root = './LLM_data'
    
    eval_graphs = {}
    for data_name in args.eval_data_names:
        pyg_data = torch.load(os.path.join(downstream_root, f'{data_name}_fixed_sbert.pt'))
        graph = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]), num_nodes=pyg_data.num_nodes)
        graph.ndata['feat'] = pyg_data.x
        graph.ndata['label'] = pyg_data.y
        if args.eval_type == 'sg':
            input_context, input_tensor = get_input(graph, args.eval_k, args.eval_length, args.eval_sgc_input_rate)
        elif args.eval_type == 'rw':
            input_context, input_tensor = get_rw_input(graph, args.eval_p, args.eval_q, args.eval_length, args.eval_sgc_input_rate)
        else:
            raise NotImplementedError
        eval_graphs[data_name] = (input_context, input_tensor, pyg_data)
        gc.collect()

    # prepare link data
    link_root = './link_data'
    link_data_all = get_link_data_all(args.eval_data_names, 0.3, 0.1, 100, 0, downstream_root, link_root)

    
    train(
        dataloader, 
        sampler,
        model,
        optimizer,
        scheduler,
        scaler,
        args.tasks, 
        args.epochs,
        rank,
        device,
        args.ckpt_path,
        eval_graphs,
        link_data_all,
        args.eval_agg,
        args.eval_p,
        args.eval_q,
        args.eval_length,
        args.eval_add_self,
        args.add_cls,
        args.alpha,
        )
    
    print("Optimization Finished!")
    print('--')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--data_size', type=int, default=0)
    # parser.add_argument('--pretrain_data_name', type=str)
    # mssl parameters
    parser.add_argument('--tasks', metavar='N', type=str, nargs='+')
    parser.add_argument('--mask_rate', type=float, default=0.2)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--p_random', type=float, default=0.2)
    parser.add_argument('--p_unchanged', type=float, default=0.2)
    parser.add_argument('--seq_batch_size', type=int, default=10000)
    parser.add_argument('--pretext', type=str, default='MLM')
    parser.add_argument('--g_aug_rate', type=float, default=0.0)
    parser.add_argument('--negative_method', type=str, default='default')
    parser.add_argument('--add_extra_negative', action='store_true')
    parser.add_argument('--undirected', action='store_true')
    parser.add_argument('--mask_center', action='store_true')

    # model parameters
    parser.add_argument('--init_method', type=str)
    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--pe_type', type=str)
    parser.add_argument('--hf_ckpt_name', type=str)
    parser.add_argument('--hf_use_pretrained', action='store_true')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--max_degree', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=384,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=1024,
                        help='FFN size')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--emb_dropout', type=float, default=0.0,
                        help='Dropout in the embedding layer')
    parser.add_argument('--emb_layer_norm', action='store_true')
    parser.add_argument('--use_degree_emb', action='store_true')
    parser.add_argument('--bin_degree', action='store_true')
    parser.add_argument('--merge_degree', action='store_true')
    parser.add_argument('--bin_base', type=int, default=2)
    parser.add_argument('--add_cls', action='store_true')
    parser.add_argument('--mlp_dims', metavar='N', type=int, nargs='+')
    parser.add_argument('--temperature', type=float, default=1.0)


    # training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=100,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--ckpt_path', type=str)

    # evaluation parameters
    parser.add_argument('--eval_data_names', metavar='N', type=str, nargs='+')
    parser.add_argument('--eval_type', type=str, default='rw')
    parser.add_argument('--eval_k', type=int, default=3)
    parser.add_argument('--eval_p', type=float, default=1.0)
    parser.add_argument('--eval_q', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=50)
    parser.add_argument('--eval_sgc_input_rate', type=float, default=0)
    parser.add_argument('--eval_agg', type=str, default='self')
    parser.add_argument('--eval_add_self', action='store_true')

    #general parameters
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--gpu_mem', type=int, default=0)

    args = parser.parse_args()
    print(args)
    num_gpus = args.num_gpus
    port = args.port
    seed = 42 
  
    mp.spawn(main, args=(num_gpus, seed, args), nprocs=num_gpus)


