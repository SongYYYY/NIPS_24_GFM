import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv
import dgl 
from dgl import GCNNorm
from torch.cuda.amp import autocast
from transformers import BertModel, DistilBertModel, AlbertModel
from transformers import AutoModel, AutoConfig
import transformers
transformers.logging.set_verbosity_info()

class GraphBertEmbeddings(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_length, max_in_degree=10000, max_out_degree=10000, merge_degree=False,
                 dropout=0, layer_norm=True, use_degree_emb=True, bin_degree=True, bin_base=2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, in_dim))
        self.embeddings = nn.Linear(in_dim, hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = layer_norm
        self.use_degree_emb = use_degree_emb
        self.bin_degree = bin_degree
        if bin_degree:
            bins, bin_edges = self.create_logarithmic_bins(max_degree=10000, base=bin_base)
            n_bins = len(bin_edges)
            self.bin_edges = bin_edges
            self.in_degree_embeddings = nn.Embedding(n_bins, hidden_dim)
            self.out_degree_embeddings = nn.Embedding(n_bins, hidden_dim)
        else:
            self.in_degree_embeddings = nn.Embedding(max_in_degree, hidden_dim)
            self.out_degree_embeddings = nn.Embedding(max_out_degree, hidden_dim)
        self.merge_degree = merge_degree

    def forward(self, pretrain=True,
                node_features=None, walks=None, 
                mask_masked=None, 
                mask_random=None,
                input_tensor=None,
                node_in_degree=None,
                node_out_degree=None,
                add_cls=False):
        
        d = self.in_dim

        if pretrain:
            b = walks.shape[0]
            node_embs = node_features
            x = node_embs[walks] # (b, l, d)
            # masking
            x[mask_masked] = self.mask_token
            x[mask_random] = node_embs[torch.randint(0, node_embs.shape[0], (mask_random.sum(),))]

            if add_cls:
                cls = self.cls_token.unsqueeze(0).expand(b, 1, d)
                x = torch.concat([cls, x], dim=1) # (b, l+1, d)

            x = self.embeddings(x)
            # pos emb
            x += self.position_embeddings[:, :x.shape[1], :]
            # degree emb
            if self.use_degree_emb:
                self.add_degree_emb(x, node_in_degree, node_out_degree, walks)
        else:
            b = input_tensor.shape[0]
            if add_cls:
                cls = self.cls_token.unsqueeze(0).expand(b, 1, d)
                x = torch.concat([cls, input_tensor], dim=1) # (b, l+1, d)
                x = self.embeddings(x)
            else:
                x = self.embeddings(input_tensor) # (b, l, d) > (b, l, d)
            # pos emb
            x += self.position_embeddings[:, :x.shape[1], :]
            # degree emb
            if self.use_degree_emb:
                self.add_degree_emb(x, node_in_degree, node_out_degree, walks)

        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.dropout(x)

        return x 
    
    def create_logarithmic_bins(self, max_degree=10000, base=2):
        n = 0
        bins = []
        current = 0

        # Continue creating bins until the upper limit of the last bin exceeds max_degree
        while current <= max_degree:
            lower_bound = current
            next_bound = int(math.pow(base, n + 1)) - 1
            upper_bound = min(next_bound, max_degree)
            bins.append((lower_bound, upper_bound))
            current = upper_bound + 1
            n += 1
        
        bin_edges = torch.tensor([b[1] for b in bins])

        return bins, bin_edges
    
    def get_bin_id_from_degree(self, node_degree):
        # Reshape degrees for broadcasting and create a comparison matrix
        expanded_degrees = node_degree.unsqueeze(1)
        comparison_matrix = expanded_degrees <= self.bin_edges.to(expanded_degrees.device)

        # Find the index of the first True in each row
        bin_indices = comparison_matrix.long().argmax(dim=1)

        return bin_indices
    
    def add_degree_emb(self, x, node_in_degree, node_out_degree, walks):
        if self.bin_degree:
            if not self.merge_degree:
                bin_ids_in = self.get_bin_id_from_degree(node_in_degree)
                x += self.in_degree_embeddings(bin_ids_in[walks])
                bin_ids_out = self.get_bin_id_from_degree(node_out_degree)
                x += self.out_degree_embeddings(bin_ids_out[walks])
            else:
                node_degree = node_in_degree + node_out_degree
                bin_ids = self.get_bin_id_from_degree(node_degree)
                x += self.in_degree_embeddings(bin_ids[walks])
        else:
            if not self.merge_degree:
                x += self.in_degree_embeddings(node_in_degree[walks])
                x += self.out_degree_embeddings(node_out_degree[walks])
            else:
                node_degree = node_in_degree + node_out_degree
                x += self.in_degree_embeddings(node_degree[walks])
            
        return 



class GraphBert(nn.Module):
    def __init__(self, embeddings, encoder, decoder, walk_length, alpha=1, p_random=0.2, p_unchanged=0.2, 
                 pretext='MLM', mask_center=False, init='all', device=None):
        super(GraphBert, self).__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.walk_length =walk_length
        self.alpha = alpha
        self.p_random = p_random
        self.p_unchanged = p_unchanged
        self.pretext = pretext
        self.mask_center = mask_center
        self.device = device

        if init == 'all':
            for m in self.modules():
                self.weights_init(m)
        elif init == 'no_encoder':
            self._init_weights()
        else:
            raise NotImplementedError

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Embedding):
            m.weight.data.normal_(std=0.02)


    def _init_weights(self):
        # Initialize weights for all modules except for the encoder
        for name, module in self.named_children():
            if name != 'encoder':  # Skip initialization for the encoder
                print(f'MODULE NAME: {name} initiliazed.')
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)  
                    if isinstance(m, nn.Bilinear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)      
                    if isinstance(m, nn.Embedding):
                        m.weight.data.normal_(std=0.02)


    def forward(self, sample, add_cls=False):
        node_features, walks, target_mask, in_degrees = sample

        loss = self.get_bert_loss(node_features.to(self.device), 
                                  walks.to(self.device),
                                  target_mask.to(self.device),
                                  in_degrees.to(self.device),
                                  add_cls)

        return loss

    # def get_bert_loss(self, node_features, walks_left, walks_right, mask_left, mask_right):
    def get_bert_loss(self, node_features, walks, target_mask, in_degrees, add_cls):
        # target_mask == walks == (b, l)
        # node_features: (n, d)

        if self.pretext == 'MLM':
            target_mask = target_mask
        elif self.pretext == 'MAE':
            target_mask = torch.zeros_like(walks, dtype=torch.bool)
            target_mask[:, 0] = True
        elif self.pretext == 'BOTH':
            mae_mask = torch.zeros_like(walks, dtype=torch.bool)
            mae_mask[:, 0] = True
            target_mask = target_mask | mae_mask
        else:
            raise NotImplementedError
        
        if self.mask_center:
            center_mask = (walks == walks[:, 0].unsqueeze(1))
            target_mask = target_mask | center_mask

        x_target = node_features[walks][target_mask].clone() # (m, d)
 
        # mask
        mask_masked, mask_random, mask_unchanged = self.get_masks(target_mask, 
                                                                  self.p_random, 
                                                                  self.p_unchanged) # (b, l)
        
        # compute degrees
        # out_degrees = torch.bincount(edges[0], minlength=node_features.shape[0])
        # in_degrees = torch.bincount(edges[1], minlength=node_features.shape[0])
        out_degrees = torch.zeros_like(in_degrees)

        with autocast(enabled=False):
            x = self.embeddings(pretrain=True,
                    node_features=node_features, 
                    walks=walks, 
                    mask_masked=mask_masked, 
                    mask_random=mask_random, 
                    node_in_degree=in_degrees,
                    node_out_degree=out_degrees,
                    add_cls=add_cls)
            
        h = self.encoder(x)
        
        if add_cls:
            h_mlm = h[:, 1:, :]
        else:
            h_mlm = h
        h_mlm = h_mlm[target_mask]
        x_pred = self.decoder(h_mlm).squeeze() # (m, d)
        loss_mlm = sce_loss(x_pred, x_target, alpha=self.alpha)

        loss_nsp = torch.tensor([0], device=self.device)

        return loss_mlm, loss_nsp
    
    def inference(self, x, walks, in_degrees, out_degrees, add_cls=False, return_dict=False):
        # x: (b, l, d)
        x = self.embeddings(pretrain=False, 
                            input_tensor=x, 
                            walks=walks,
                            node_in_degree=in_degrees,
                            node_out_degree=out_degrees,
                            add_cls=add_cls)

        h = self.encoder(x, return_dict=return_dict)

        return h 

     
    
    def get_masks(self, mask, p_random, p_unchanged):
        # mask: (b, l)
        assert 0 <= p_random <= 1 and 0 <= p_unchanged <= 1 and p_random + p_unchanged <= 1
    
        # Get the indices of tokens to be masked
        mask_indices = mask.nonzero(as_tuple=True)

        # Generate random decisions for each masked token
        mask_decisions = torch.rand(mask_indices[0].shape[0], device=mask.device)  # Shape: (number of masked tokens,)

        # Initialize masks for each operation
        mask_random = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # Shape: (b, l)
        mask_random[mask_indices] = mask_decisions < p_random
        
        mask_unchanged = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # Shape: (b, l)
        mask_unchanged[mask_indices] = (mask_decisions >= p_random) & (mask_decisions < p_random + p_unchanged)
        
        # The remaining tokens are masked with [MASK]
        mask_masked = mask.clone()  # Shape: (b, l)
        mask_masked[mask_random] = False
        mask_masked[mask_unchanged] = False

        return mask_masked, mask_random, mask_unchanged

    def get_sgc_feat(self, x, edges):
        num_nodes = x.shape[0]
        self_loops = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1).to(self.device)
        edges = torch.cat([edges, self_loops], dim=1)
        graph = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
        graph = self.transform(graph)
        edge_weights = graph.edata['w']
        adj = torch.sparse_coo_tensor(edges, edge_weights, (num_nodes, num_nodes))
        z = torch.sparse.mm(adj, x)

        return z
    
    def get_nsp_loss(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]

        if self.nsp_sim_type == 'sigmoid':
            d_norm = np.sqrt(hidden1.shape[1])
            logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1) / d_norm)
            logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1) / d_norm)
            logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1) / d_norm)
            logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1) / d_norm)
        elif self.nsp_sim_type == 'cos':
            logits_aa = F.cosine_similarity(hidden1, summary1, dim=1)
            logits_bb = F.cosine_similarity(hidden2, summary2, dim=1)
            logits_ab = F.cosine_similarity(hidden1, summary2, dim=1)
            logits_ba = F.cosine_similarity(hidden2, summary1, dim=1)
        else:
            raise NotImplementedError
        
        # print(f'aa: {logits_aa}')
        # print(f'ba: {logits_ba}')
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        # print(f'NSP LOSS 1: {TotalLoss.item()}')
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        # print(f'NSP LOSS 2: {TotalLoss.item()}')
        return TotalLoss
        


        

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class MLPDecoder(nn.Module):
    def __init__(self, layer_sizes, bias=False):
        """
        Initialize the MLP.
        
        Parameters:
        layer_sizes (list): A list containing the sizes of each layer. The first element should be the input size,
                            and the last element should be the output size. Intermediate elements are the sizes of
                            the hidden layers.
        """
        super(MLPDecoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Create the hidden layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
            
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Parameters:
        x (Tensor): The input tensor.
        
        Returns:
        Tensor: The output of the MLP.
        """
        # Pass the input through each layer followed by a ReLU activation function, except for the last layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function for all but the last layer
                x = F.relu(x)
        
        return x
    

class BertEncoder(nn.Module):
    def __init__(self, ckpt_name, n_layers, use_pretrained=True) -> None:
        super().__init__()
        if use_pretrained:
            model = AutoModel.from_pretrained(ckpt_name)
        else:
            config = AutoConfig.from_pretrained(ckpt_name)
            model = AutoModel.from_config(config)

        model.encoder.layer = model.encoder.layer[:n_layers]
        self.encoder = model.encoder

    def forward(self, x, return_dict=False):
        if return_dict:
            return self.encoder(x, output_hidden_states=True)
        else:
            return self.encoder(x)[0]
        
class SymmetricScoringFunction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class AsymmetricScoringFunction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels*2, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels*2, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)



class ContrastiveScoringFunction(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=0)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        s = (x[:x_i.shape[0]] * x[x_i.shape[0]:]).sum(dim=-1, keepdim=True)
        return s 
        

class IdentityMapping(nn.Module):
    def __init__(self):
        super(IdentityMapping, self).__init__()
        
    def forward(self, x):
        return x
        

