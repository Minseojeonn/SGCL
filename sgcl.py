import math
import os
from collections import defaultdict
from itertools import product
import random
from dotmap import DotMap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
torch.multiprocessing.set_sharing_strategy('file_system')
import dgl
import dgl.nn
import dgl.function as fn
from dgl.nn import RelGraphConv
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
import pickle
import tqdm as tqdm
import warnings
from sklearn.model_selection import train_test_split
from utils import set_random_seed, split_data, load_data
warnings.filterwarnings("ignore")
seed = 100
dataset_name = 'bonanza'
input_dim = 16
device = 'cuda:0'
############################################### Load Dataset ############################################################
set_random_seed(seed, device)
path = 'datasets/' + dataset_name + '.tsv'
edgelist, num_nodes, num_edges = load_data(path, True, "uni")
train_X, train_Y, val_X, val_Y, test_X, test_Y = split_data(edgelist, [0.85, 0.05, 0.1], seed, True)
df = pd.DataFrame(edgelist, columns=['src', 'dst', 'label'])

df_train = pd.DataFrame(np.concatenate([train_X, train_Y.reshape(-1, 1)], axis=1), columns=['src', 'dst', 'label'])
df_val = pd.DataFrame(np.concatenate([val_X, val_Y.reshape(-1, 1)], axis=1), columns=['src', 'dst', 'label'])
df_test = pd.DataFrame(np.concatenate([test_X, test_Y.reshape(-1, 1)], axis=1), columns=['src', 'dst', 'label'])
############################################### Load user and features ############################################################
het_num_nodes_dict = {}
het_node_feat_dict = {}
het_data_dict = {}
het_edge_feat_dict = {}

het_num_nodes_dict['user'] = sum(num_nodes)
user_feat = np.random.rand(sum(num_nodes), input_dim)
het_node_feat_dict['user'] = user_feat
    
############################################### Load Train Graph ############################################################


with open(f"./bitcoin_alpha/g_train.pkl", "rb") as f:
    tmp_het_data_dict = pickle.load(f)
    tmp_het_edata_dict2 = tmp_het_data_dict
    
train_positive = train_X[train_Y==1]
train_negative = train_X[train_Y==0].T
tmp_het_data_dict = {('user', 'positive', 'user'): (train_positive[0], train_positive[1]), ('user', 'negative', 'user'): (train_negative[0], train_negative[1])}
het_data_dict.update(tmp_het_data_dict)

graph_user = dgl.heterograph(
    data_dict = het_data_dict,
    num_nodes_dict = het_num_nodes_dict
)
for node_t in het_node_feat_dict:
    graph_user.nodes[node_t].data['feature'] = torch.from_numpy(het_node_feat_dict[node_t]).float()

for i in tmp_het_data_dict.items():
  print('items')
  print(i)

############################################### Load Labels ############################################################
class LabelPairs(torch.utils.data.Dataset):
    def __init__(self, df):
        super(LabelPairs).__init__()
        u = torch.from_numpy(df.src.values).long()
        v = torch.from_numpy(df.dst.values).long()
        y = torch.from_numpy(df['label'].values).double()
        self.pairs = torch.stack((u, v), dim=0)
        self.label = y
    
    def __getitem__(self, index):
        return self.pairs[:, index], self.label[index]
    
    def __len__(self):
        return len(self.label)

class NodeBatch(torch.utils.data.Dataset):
    def __init__(self, nodes):
        self.nodes = torch.from_numpy(nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __len__(self):
        return len(self.nodes)



"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import torch
import torch as th
from torch import nn
from torch.nn import init
from torch.functional import F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils

'''
# The GraphConv class is defined as a subclass of nn.Module. It takes several parameters as inputs:

# in_feats: The number of input features per node.
# out_feats: The number of output features per node.
# norm: A string indicating the type of normalization to use. It can take one of the following values: "none", "both", or "right".
# weight: A boolean indicating whether to use a learnable weight matrix.
# bias: A boolean indicating whether to use a learnable bias vector.
# activation: An activation function to apply to the output tensor. If None, no activation is applied.
# residual: A boolean indicating whether to use residual connections.
# allow_zero_in_degree: A boolean indicating whether to allow nodes with zero in-degree in the graph.
# The __init__ method initializes the parameters of the GraphConv module, including the weight and bias parameters, and sets the activation function and normalization type.

# The reset_parameters method initializes the weight and bias parameters of the module using the Xavier initialization and zeros initialization, respectively.

# The set_allow_zero_in_degree method sets the allow_zero_in_degree parameter of the module.

# The forward method implements the forward pass of the GraphConv module. It takes as input a graph (graph), a feature tensor (feat), and an optional weight matrix (weight). It applies the graph convolution operation to the feature tensor and returns the output tensor.

# Inside the forward method, the graph.local_scope() context manager is used to create a local scope for any changes made to the graph.

# If allow_zero_in_degree is set to False, the method checks whether the graph contains nodes with zero in-degree. If so, it raises a DGLError.

# If the norm parameter is set to "both", it normalizes the feature tensor using the degree of the outgoing edges of each node in the graph.

# If residual is set to True, it adds a self-loop to the graph and applies a linear transformation to the feature tensor to compute a residual connection.

# If weight is not provided, it uses the weight parameter of the module. If in_feats is greater than out_feats, it applies the weight matrix to the feature tensor before aggregating the neighbors' features. Otherwise, it aggregates the neighbors' features before applying the weight matrix.

# If the norm parameter is set to "none", it normalizes the output tensor by dividing it by the in-degree of each node in the graph. If it is set to "both", it multiplies the output tensor by the square root of the inverse of the in-degree and the out-degree of each node in the graph.

# If bias is set to True, it adds the bias vector to the output tensor.

# If activation is provided, it applies the activation function to the output tensor.

# If residual is set to True, it adds the residual connection to the output tensor.

# Finally, it returns the output tensor. '''
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='right',
                 weight=True,
                 bias=True,
                 activation=None,
                 residual=True,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
#         if norm not in ('none', 'both', 'right'):
        if norm not in ('right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._residual = residual
        
        if self._residual:
            self.loop_weight = nn.Linear(in_feats, out_feats, bias=False)
        
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):

        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if self._residual:
                loop_message = self.loop_weight(feat_dst)
            if graph.num_edges() == 0:
                return loop_message                
                
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
                
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)
                
            if self._residual:
                rst = rst + loop_message

            return rst

'''
 The GATConv module takes as input a graph and node feature tensor and produces an output feature tensor of the same shape. The constructor for the module takes the following arguments:

in_feats: the number of input node features.
out_feats: the number of output node features.
num_heads: the number of attention heads to use.
feat_drop: the probability of dropping out input node features during training.
attn_drop: the probability of dropping out attention weights during training.
negative_slope: the slope of the leaky ReLU activation function used in computing the attention coefficients.
residual: whether to include a residual connection in the module.
activation: the activation function to use in the module.
allow_zero_in_degree: whether to allow nodes in the input graph with zero in-degree.
The forward method of the module takes a graph and node feature tensor as input and produces an output feature tensor by performing the following steps:

Apply dropout to the input node features.
Compute the linear projections of the input node features for each attention head.
Compute the attention coefficients for each edge in the graph using the dot product of the linear projections of the source and destination nodes for each attention head.
Apply the attention coefficients to the source node features to compute the message passed to the destination nodes for each attention head.
Concatenate the messages from all attention heads and sum them to produce the output node features.
Apply activation and/or residual connection if specified.
Overall, this code provides an implementation of the GATConv module that can be used in building GAT-based GNNs for graph classification, node classification, and other graph-related tasks. 

'''    
    
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
                
            rst = rst.view(len(rst), -1)
                
            return rst
    

    
''' 
The HeteroGraphConv class takes several inputs: mods, a dictionary of modules for each edge type; dim_values, the dimension of the input node embeddings; dim_query, the dimension of the attention query; and agg_type, the type of aggregation function to use during message passing.

The constructor initializes the modules for each edge type in mods, and sets the input dimensions and attention layers for each module. 
If the agg_type is specified, the corresponding aggregation function is also set.

The forward method takes as input a graph g, a set of node embeddings inputs, and optional arguments for the edge type modules. 
The method first checks whether the input is a tuple or a block diagonal graph, and assigns the source and destination node embeddings accordingly. 
It then loops through each canonical edge type in the graph, and computes the output embeddings and attention scores for each type. 
If an edge type has no edges, the output and attention scores are set to zero.

After processing all edge types, the method aggregates the output embeddings based on the agg_type specified during initialization. 
If the agg_type is attention-based, the attention scores are used to weight the output embeddings. Finally, the method returns the aggregated output embeddings and the attention scores.
''' 
    
class HeteroGraphConv(nn.Module):
    def __init__(self, mods, dim_values, dim_query, agg_type='attn'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
                
        self.dim_values = dim_values
        self.dim_query = dim_query
        
        self.attention = nn.ModuleDict()
        for k, _ in self.mods.items():
            self.attention[k] = nn.Sequential(nn.Linear(dim_values, dim_query), nn.Tanh(), nn.Linear(dim_query, 1, bias=False))
        
        self.agg_type = agg_type
        if agg_type == 'sum':
            self.agg_fn = th.sum
        elif agg_type == 'max':
            self.agg_fn = lambda inputs, dim: th.max(inputs, dim=dim)[0]
        elif agg_type == 'min':
            self.agg_fn = lambda inputs, dim: th.min(inputs, dim=dim)[0]
        elif agg_type == 'stack':
            self.agg_fn = th.stack
        
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = []
        et_scores = []
        et_count = 0
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = inputs[:g.number_of_dst_nodes()]

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    et_scores.append(torch.zeros(g.number_of_dst_nodes(), 1).to(g.device))
                    outputs.append(torch.zeros(g.number_of_dst_nodes(), self.dim_values).to(g.device))
                    continue
                et_count += 1
                dstdata = self.mods[etype](rel_graph, (src_inputs, dst_inputs))
                outputs.append(dstdata)
                et_scores.append(self.attention[etype](dstdata))
        if len(outputs) == 0:
            out_embs = dst_inputs
        else:
            et_dst_data = torch.stack(outputs, dim=0)
            if self.agg_type == 'attn':
                attn = torch.softmax(torch.stack(et_scores, dim=0), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'attn_sum':
                attn = torch.softmax(torch.stack(et_scores, dim=0).mean(dim=1, keepdims=True), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'mean':
                out_embs = torch.sum(torch.stack(et_scores, dim=0), dim=0) / et_count
            else:
                out_embs = self.agg_fn(et_dst_data, dim=0)
        return out_embs, attn    

############################################## Define Model ############################################################
"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name

from torch import nn
from torch.nn import init
from torch.functional import F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils


'''
Define Hetero Layer
'''
'''
This is a PyTorch implementation of a heterogeneous graph neural network called HetAttn. 
The code contains two functions: calc_from_loader and forward. 
The calc_from_loader function takes in a data loader, node features, and a device and returns the node embeddings and attention results. 
The forward function takes in a graph, node features, node IDs, and a device and returns the node embeddings and attention results for the specified nodes.
'''

'''
The HetAttn class is defined as a subclass of nn.Module and takes in two arguments: args and etypes. The args argument is a dictionary of hyperparameters and the etypes argument is a list of edge types in the graph.

The __init__ function initializes several instance variables.
The args argument is converted to a DotMap object and saved as an instance variable.
The etypes argument is saved as an instance variable. 
A linear layer called feature_trans is defined that takes in node features of size args.dim_features and outputs node embeddings of size args.dim_hiddens. 
A nn.ModuleList called convs is initialized to hold the graph convolutional layers that will be added later.
 
'''
'''
This is a PyTorch module named HetAttn that implements a heterogeneous graph neural network (GNN) with attention mechanism. 
The __init__ method initializes the model's parameters, including linear layers and graph convolutional layers (GraphConv or GATConv). 
The calc_from_loader method calculates the embeddings for a batch of nodes, and the forward method applies the model to a graph and returns the embeddings for the specified nodes. 
The inference method is similar to the forward method, but uses a different sampler for better inference performance.

The HetAttn class takes two arguments in its constructor: args and etypes. args is a dictionary-like object containing various hyperparameters of the model, while etypes is a list of the edge types in the heterogeneous graph.

The feature_trans linear layer maps the input features to a higher-dimensional space. 
The convs list contains multiple HeteroGraphConv layers, each of which applies a graph convolution operation to the input. 
The specific type of graph convolution used depends on the value of args.conv_type. 
If args.conv_type is 'gcn', GraphConv is used, while if it is 'gat', GATConv is used. 
The number of graph convolution layers is controlled by args.conv_depth. 
The output of each graph convolution layer is concatenated with the previous layers' outputs and passed through another linear layer (concat_weight) to obtain the final node embeddings.

The calc_from_loader method takes a loader object, which generates node pairs and their corresponding subgraphs. 
It iteratively applies the HeteroGraphConv layers to the input features and concatenates the resulting embeddings to obtain the final node embeddings.

The forward and inference methods are similar, except for the type of neighbor sampler used. 
They both take a graph g, a feature matrix x, a list of node IDs nids, and a device on which to perform computations. 
They return the node embeddings for the specified nodes nids and the attention weights used during computation.
'''
class HetAttn(nn.Module):
    def __init__(self, args, etypes):
        super(HetAttn, self).__init__()
        self.args = DotMap(args.toDict())
        args = self.args
        self.etypes = etypes
        self.feature_trans = nn.Linear(args.dim_features, args.dim_hiddens, bias=False)
        self.convs = nn.ModuleList()
        
        if self.args.conv_type == 'gcn':
            for _ in range(args.conv_depth - 1):
                self.convs.append(HeteroGraphConv({rel: GraphConv(args.dim_hiddens, args.dim_hiddens, allow_zero_in_degree=True, residual=args.residual)
                                                   for rel in self.etypes},
                                  agg_type=args.het_agg_type, dim_values=args.dim_hiddens, dim_query=args.dim_query)) 
            self.convs.append(HeteroGraphConv({rel: GraphConv(args.dim_hiddens, args.dim_embs, allow_zero_in_degree=True, residual=args.residual) 
                                               for rel in self.etypes},
                              agg_type=args.het_agg_type, dim_values=args.dim_embs, dim_query=args.dim_query))
        elif self.args.conv_type == 'gat':
            for _ in range(args.conv_depth - 1):
                self.convs.append(HeteroGraphConv({rel: GATConv(args.dim_hiddens, args.dim_hiddens // args.num_heads, args.num_heads, allow_zero_in_degree=True, residual=args.residual)
                                                   for rel in self.etypes},
                                  agg_type=args.het_agg_type, dim_values=args.dim_hiddens, dim_query=args.dim_query)) 
            self.convs.append(HeteroGraphConv({rel: GATConv(args.dim_hiddens, args.dim_embs // args.num_heads, args.num_heads, allow_zero_in_degree=True, residual=args.residual) 
                                               for rel in self.etypes},
                          agg_type=args.het_agg_type, dim_values=args.dim_embs, dim_query=args.dim_query))
        
        self.concat_weight = nn.Linear((args.conv_depth + 1) * args.dim_embs, args.dim_embs, bias=False)

        if self.args.conv_depth == 1:
            self.sampler_nodes = [5]
            self.sampler_inference = [10]
        elif self.args.conv_depth == 2:
            self.sampler_nodes = [10, 20]
            self.sampler_inference = [10 , 20]
        else:
            raise
#         if self.args.conv_depth == 3:
#             self.sampler_nodes = [5, 10, 10]
#             self.sampler_inference = [10, 10, 10]
            
        nn.init.xavier_uniform_(self.feature_trans.weight)
        
        
    def calc_from_loader(self, loader, x, device):
        y = torch.zeros(len(x), self.args.dim_embs)
        attn_res = torch.zeros(len(x), len(self.etypes))
        
        def calc_from_blocks(blocks, conv_idx, x, device):
            input_nodes, output_nodes = blocks[0].srcdata[dgl.NID], blocks[0].dstdata[dgl.NID]
            h = x[input_nodes].to(device)
            h = torch.tanh(self.feature_trans(h))
            for b, idx in zip(blocks, conv_idx):
                b = b.to(device)
                h, attn = self.convs[idx](b, h)
            return h, attn
        
        for input_nodes, output_nodes, blocks in loader:
            
            h0 = x[output_nodes].to(device)
            h0 = torch.tanh(self.feature_trans(h0))
            emb_ulti = [h0]
            if self.args.conv_depth == 1:
                h1, attn = calc_from_blocks(blocks, [0], x, device)
                emb_ulti.append(h1)
            if self.args.conv_depth ==2:
                h1, _ = calc_from_blocks(blocks[1:], [1], x, device)
                h2, attn = calc_from_blocks(blocks, [0, 1], x, device)
                emb_ulti.extend([h1, h2])
            
            y[output_nodes] = self.concat_weight(torch.cat(emb_ulti, dim=-1)).cpu()
            attn_res[output_nodes] = attn.squeeze(dim=-1).transpose(0, 1).cpu()
        return y, attn_res

    def forward(self, g, x, nids, device):
        dataloader = dgl.dataloading.DataLoader(g, nids,
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_nodes),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        y, attn_res = self.calc_from_loader(dataloader, x, device)
        return y, attn_res
    
    
    def inference(self, g, x, nids, device):
#         dataloader = dgl.dataloading.NodeDataLoader(g, nids,
#                                                     dgl.dataloading.MultiLayerFullNeighborSampler(len(self.sampler_nodes)),
#                                                     batch_size=self.args.inference_batch_size,
#                                                     num_workers=self.args.num_workers,
#                                                     shuffle=True,
#                                                     drop_last=False)
        dataloader = dgl.dataloading.DataLoader(g, nids,
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_inference),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        y, attn_res = self.calc_from_loader(dataloader, x, device)
        return y, attn_res
    

'''
Define Whole Model
'''
'''
This is a PyTorch module definition with a class name Model. 
The module represents a neural network architecture. 
The class has four methods: __init__(), forward(), inference(), and compute_contrastive_loss().

__init__() method initializes the class by setting up its parameters and neural network architecture.

forward() method implements the forward pass of the neural network. 
It takes four input parameters: g_attr, g_stru, nids, and device. g_attr and g_stru are graph structures with node and edge features. nids is a tensor containing node ids. device is the device on which the computations are performed. The method computes the node embeddings for both graphs and returns them.

inference() method is used to generate the embeddings of the nodes using the trained model. 
It is similar to the forward() method, but it returns the attention scores in addition to the node embeddings.

compute_contrastive_loss() method is used to compute the contrastive loss between the embeddings of the positive and negative samples. 
The method takes the device and node embeddings as inputs and returns the contrastive loss.

The model has a ScorePredictor class which predicts the similarity score between two given nodes based on their node embeddings. 
The HetAttn class computes the node embeddings using heterogeneous attention mechanism. The embeddings of nodes in g_attr and g_stru are computed separately using the pos_emb_model and neg_emb_model if args.sign_conv equals 'sign'. Otherwise, only one emb_model is used. 
The transform and attention linear layers are used to combine the embeddings of nodes.

The model also has some other parameters like dim_embs, combine_type, and sign_aggre, which are used to define the neural network architecture.
'''
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.sign_conv == 'sign':
            self.pos_emb_model = HetAttn(args, args.pos_edge_type)
            self.neg_emb_model = HetAttn(args, args.neg_edge_type)
        elif args.sign_conv == 'common':
            self.emb_model = HetAttn(args, args.edge_type)
        self.args = args
        self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)
        
        self.combine_type = args.combine_type
        
        if self.args.sign_aggre!='both':
            transform_type = 2
        elif self.args.sign_aggre == 'both' or self.args.sign_conv == 'common':
            transform_type = 4
        
        if self.combine_type == 'concat':
            self.transform = nn.Sequential(nn.Linear(transform_type*args.dim_embs, args.dim_embs))   #transform
            self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)
        elif self.combine_type == 'attn':
            self.attention = nn.Sequential(nn.Linear(args.dim_embs, args.dim_query), nn.Tanh(), nn.Linear(args.dim_query, 1, bias=False))
            self.link_predictor = ScorePredictor(args, dim_embs=args.dim_embs)
        
    def forward(self, g_attr, g_stru, nids, device):
#         nids = torch.unique(torch.cat((uids, vids), dim=-1))
#         embs_pos, embs_neg = self.emb_model(g, x, nids, device)
#         score = self.predict_combine((embs_pos, embs_neg), uids, vids, device)
        if self.args.sign_conv == 'common':
            embs_attr_pos, _ = self.emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, _ = self.emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_attr_neg, _ = self.emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_neg, _ = self.emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            return embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg
        elif self.args.sign_conv == 'sign':
            embs_attr_pos, _ = self.pos_emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_attr_neg, _ = self.neg_emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, _ = self.pos_emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_stru_neg, _ = self.neg_emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            return embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg
    
    def inference(self, g_attr, g_stru, nids, device):
        if self.args.sign_conv == 'common':
            embs_attr_pos, attn_attr_pos = self.emb_model(g_attr, g_attr.ndata['feature'], nids, device)
            embs_attr_neg, attn_attr_neg = self.emb_model(g_attr, g_attr.ndata['feature'], nids, device)
            embs_stru_pos, attn_stru_pos = self.emb_model(g_stru, g_stru.ndata['feature'], nids, device)
            embs_stru_neg, attn_stru_neg = self.emb_model(g_stru, g_stru.ndata['feature'], nids, device)
            return (embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg), (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg)
        elif self.args.sign_conv == 'sign':
            embs_attr_pos, attn_attr_pos = self.pos_emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_attr_neg, attn_attr_neg = self.neg_emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, attn_stru_pos = self.pos_emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_stru_neg, attn_stru_neg = self.neg_emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            print()
            return (embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg), (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg)

        
    def compute_contrastive_loss(self, device, embs_attr_pos, embs_stru_pos, embs_attr_neg=None, embs_stru_neg=None):
        nodes_num = embs_attr_pos.shape[0]
        feature_size = embs_attr_pos.shape[1]
        
        embs_attr_pos = embs_attr_pos.to(device)
        embs_stru_pos = embs_stru_pos.to(device)
        normalized_embs_attr_pos = F.normalize(embs_attr_pos, p=2, dim=1)
        normalized_embs_stru_pos = F.normalize(embs_stru_pos, p=2, dim=1)
        if embs_attr_neg!=None and embs_stru_neg!=None:
            embs_attr_neg = embs_attr_neg.to(device)
            embs_stru_neg = embs_stru_neg.to(device)
            normalized_embs_attr_neg = F.normalize(embs_attr_neg, p=2, dim=1)
            normalized_embs_stru_neg = F.normalize(embs_stru_neg, p=2, dim=1)
        
        
        def inter_contrastive(embs_attr, embs_stru):
            pos = torch.exp(torch.div(torch.bmm(embs_attr.view(nodes_num, 1, feature_size), embs_stru.view(nodes_num, feature_size, 1)), self.args.tao))
            
            def generate_neg_score(embs_1, embs_2):
                neg_similarity = torch.mm(embs_1.view(nodes_num, feature_size), embs_2.transpose(0,1))
                neg_similarity[np.arange(nodes_num),np.arange(nodes_num)] = 0
                return torch.sum(torch.exp(torch.div( neg_similarity  , self.args.tao)) , dim=1)
            
            neg = generate_neg_score(embs_attr, embs_stru)

            return torch.mean(- (torch.log(torch.div(pos, neg))))
        
        def intra_contrastive(self_embs, embs_attr_pos, embs_attr_neg, embs_stru_pos, embs_stru_neg):
            pos_score_1 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_attr_pos.view(nodes_num, feature_size, 1)), self.args.tao))
            pos_score_2 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_stru_pos.view(nodes_num, feature_size, 1)), self.args.tao))
            pos = pos_score_1 + pos_score_2
            def generate_neg_score(pos_embs, neg_embs_1, neg_embs_2):
                neg_score_1 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_1.view(nodes_num, feature_size, 1))
                neg_score_2 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_2.view(nodes_num, feature_size, 1))
                return torch.exp(torch.div(neg_score_1, self.args.tao)) + torch.exp(torch.div(neg_score_2, self.args.tao))
            neg = generate_neg_score(self_embs, embs_attr_neg, embs_stru_neg)
            return torch.mean(- torch.log(torch.div(pos, neg)) )
            

        inter_pos = inter_contrastive(normalized_embs_attr_pos, normalized_embs_stru_pos)
        inter_neg = inter_contrastive(normalized_embs_attr_neg, normalized_embs_stru_neg)
        
        embs = torch.cat((embs_attr_pos,embs_stru_pos,embs_attr_neg, embs_stru_neg), dim=-1)
        # print("#####################################")
        # print("embs_attr_pos")
        # print(embs_attr_pos.size())
        # print("embs_stru_pos")
        # print(embs_stru_pos.size())
        # print("embs_attr_neg")
        # print(embs_attr_neg.size())
        # print("embs_stru_neg")
        # print(embs_stru_neg.size())
   
        # print("concatenated emb of ultimate representation")
        # print(embs.size())
        self_embs = self.transform(embs)
        # print("transformed emb of ultimate representation ")
        # print(self_embs.size())
        normalized_self_embs = F.normalize(self_embs, p=2, dim=1)
        # print("normalized emb of ultimate representation ")
        # print(normalized_self_embs.size())

        
        intra = intra_contrastive(normalized_self_embs, normalized_embs_attr_pos, normalized_embs_attr_neg, normalized_embs_stru_pos, normalized_embs_stru_neg)
        # print(f'inter_pos:{inter_pos}  inter_neg:{inter_neg}  intra:{intra}')
        if self.args.contrast_type == 'pos':
            return inter_pos
        elif self.args.contrast_type == 'neg':
            return inter_neg
        elif self.args.contrast_type == 'intra':
            return intra
        elif self.args.contrast_type == 'inter':
            return inter_pos + inter_neg
        elif self.args.contrast_type == 'all':
            return (1-self.args.beta) * (inter_pos + inter_neg) + self.args.beta * intra
            

        
    
    def compute_label_loss(self, score, y_label, pos_weight, device):
        pos_weight = torch.tensor([(y_label==0).sum().item()/(y_label==1).sum().item()]*y_label.shape[0]).to(device)
        return F.binary_cross_entropy_with_logits(score, y_label, pos_weight=pos_weight)
    
        
        
    def predict_combine(self, embs, uids, vids, device):
        u_embs = self.combine(embs, uids, device)
        v_embs = self.combine(embs, vids, device)
        score = self.link_predictor(u_embs, v_embs)
        return score
    
    def compute_attention(self, embs):
        attn = self.attention(embs).softmax(dim=0)
        return attn
    
    def combine(self, embs, nids, device):
        if self.args.sign_conv == 'sign':
            if self.args.sign_aggre == 'pos':
                embs = (embs[0],embs[1])
            elif self.args.sign_aggre == 'neg':
                embs = (embs[2],embs[3])
            
        if self.combine_type == 'concat':
            embs = torch.cat(embs, dim=-1)
            sub_embs = embs[nids].to(device)
            out_embs = self.transform(sub_embs)
            return out_embs                          #output embs
        elif self.combine_type == 'attn':
            embs = torch.stack(embs, dim=0)
            sub_embs = embs[:,nids].to(device)
            attn = self.compute_attention(sub_embs)
            # attn: (2,n,1)   sub_embs: (2,n,feature)
            out_embs = (attn*sub_embs).sum(dim=0)
            return out_embs
        elif self.combine_type == 'mean':
            embs = torch.stack(embs, dim=0).mean(dim=0)
            sub_embs = embs[nids].to(device)
            return sub_embs
        elif self.combine_type == 'pos':
            sub_embs = embs[0][nids].to(device)
            return sub_embs



############################################### Graph Augmentation ############################################################
"""
This code contains a series of functions that augment a graph by adding noise. 
The code is written in Python and uses the DGL (Deep Graph Library) library for manipulating the graph.
"""
"""
This generate_mask function generates a mask for the graph. 
The mask_ratio parameter determines the proportion of the mask that is filled with 0s (elements to drop) or 1s (elements to leave). 
The row and column parameters specify the dimensions of the mask. 
The function generates a random array of floats between 0 and 1 of the specified dimensions, masks the values below the mask_ratio parameter with 0s, and the rest with 1s.

"""
def generate_mask(mask_ratio, row, column):
    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0,1,size=(row, column))
    arr_mask = np.ma.masked_array(arr_mask_ratio, mask=(arr_mask_ratio<mask_ratio)).filled(0)
    arr_mask = np.ma.masked_array(arr_mask, mask=(arr_mask>=mask_ratio)).filled(1)
    return arr_mask

"""
generate_attr_graph function generates noise for the graph features (node attributes) by adding random noise and dropping some elements using the mask generated by generate_mask(). 
The g parameter is the input graph, and args is a collection of arguments for the augmentation. 
The function generates random noise by sampling from a normal distribution with a mean of 0 and a standard deviation of 0.1. 
Then it applies the mask to the feature matrix, element-wise, and adds the noise to the remaining elements. 
The function returns a new graph with the noisy features.
"""    

def generate_attr_graph(g, args):
    # generate noise g_attr
    feature = g.ndata['feature']
    attr_noise = np.random.normal(loc=0, scale=0.1, size=(feature.shape[0], feature.shape[1]))
    attr_mask = generate_mask(args.mask_ratio, row=feature.shape[0], column=feature.shape[1])
    noise_feature = feature*attr_mask + (1-attr_mask) * attr_noise
    
    g_attr = g
    g_attr.ndata['feature'] = noise_feature.float()
    return g_attr

"""
function generate_stru_graph takes a graph g and some arguments args. 
It creates a copy of the graph g_stru and deletes a certain percentage of edges of specific types specified in args. It then adds an equal number of randomly sampled edges back into the graph, ensuring that they don't already exist in the original graph.
Finally, it casts the node features to float and returns the augmented graph.
"""
def generate_stru_graph(g, args):
    # generate noise g_stru by deleting links
    g_stru = g

    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
        
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        # shape: (e, 2)
        df = np.array([etype_edges[0].numpy(), etype_edges[1].numpy()]).transpose()
        
        # delete edges
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0])).squeeze()
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)

        # add an equal number of edges
        add_row = []
        add_column = []
        index = 0
        while index < len(drop_eids):
            row_sample = np.random.randint(g.num_nodes())
            column_sample = np.random.randint(g.num_nodes())
            if (df==[row_sample, column_sample]).all(1).any() == False:
                index += 1
                add_row.append(row_sample)
                add_column.append(column_sample)
        g_stru = dgl.add_edges(g_stru, add_row, add_column, etype=etype)

    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru

"""
 function generate_stru_sign_graph is similar to the first, but instead of adding new edges, it exchanges some positive or negative edges with randomly selected edges of the opposite sign.
"""
def generate_stru_sign_graph(g, args):
    # generate noise g_stru by exchanging some pos/neg links
    g_stru = g
    
    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
    
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0])).squeeze()
        
        # delete edges
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)
        
        # add_edges
        if etype in args.pos_edge_type:
            g_stru = dgl.add_edges(g_stru, etype_edges[0][drop_eids], etype_edges[1][drop_eids] , etype=random.choice(args.neg_edge_type))
        elif etype in args.neg_edge_type:
            g_stru = dgl.add_edges(g_stru, etype_edges[0][drop_eids], etype_edges[1][drop_eids] , etype=random.choice(args.pos_edge_type))
    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru
"""
 function generate_stru_status_graph deletes edges of specific types as before, but instead of adding new edges, it adds reverse edges in their place.
"""
def generate_stru_status_graph(g, args):
    g_stru = g
    
    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
    
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0]))
        
        # delete edges
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)
        
        # add reverse_edges
        g_stru = dgl.add_edges(g_stru, etype_edges[1][drop_eids], etype_edges[0][drop_eids], etype=etype)
    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru
"""
function GraphAug takes a graph g and some arguments args. 
It calls one of the above functions based on the specified args.augment parameter and returns two augmented graphs, one with attribute perturbations g_attr and one with structural perturbations g_stru.

If args.augment is 'delete', it calls generate_stru_graph twice. 
If args.augment is 'change', it calls generate_stru_sign_graph twice. 
If args.augment is 'reverse', it calls generate_stru_status_graph twice. 
If args.augment is 'composite', it calls generate_stru_sign_graph once and generate_stru_graph once, returning one graph with sign perturbations and one with attribute perturbations.
"""
def GraphAug(g, args):
    if args.augment == 'delete':     #for connectivity perturbation
        g_attr = generate_stru_graph(g, args)
        g_stru = generate_stru_graph(g, args)
    elif args.augment == 'change':          #for sign perturbation
        g_attr = generate_stru_sign_graph(g, args)
        g_stru = generate_stru_sign_graph(g, args)
    elif args.augment == 'reverse':
        g_attr = generate_stru_status_graph(g, args)
        g_stru = generate_stru_status_graph(g, args)
    elif args.augment == 'composite':
        g_attr = generate_stru_sign_graph(g, args)
        g_stru = generate_stru_graph(g, args)
    return g_attr, g_stru


############################################### Define Predictor ############################################################
"""
This code defines a Python class named ScorePredictor, which is a subclass of the PyTorch nn.Module class. 
The class is used for predicting a score given two embeddings u_e and u_v. 
The class constructor takes two parameters - args, which is an instance of the DotMap class, and params, which is a dictionary containing additional parameters. 
The constructor initializes the instance variables and sets them to the DotMap object created from args.toDict().

The constructor also contains a conditional block that sets the predictor instance variable depending on the value of args.predictor. 
If args.predictor is 'dot', then predictor is set to None. 
Otherwise, predictor is set to a PyTorch sequential neural network consisting of one or more linear layers with leaky ReLU activation. 
The number of linear layers and the number of hidden units in each linear layer are determined by args.predictor.

The class contains a method named reset_parameters, which does nothing. 
The forward method takes u_e and u_v embeddings as input and returns the predicted score based on the predictor instance variable.

The next method defined is eval_model. 
This method takes four parameters - embs, which are the embeddings, model, which is an instance of the ScorePredictor class, df, which is a Pandas DataFrame, and batched, which is a boolean value indicating whether to use batching or not. 
The method creates a PyTorch DataLoader object to iterate over the input data, and returns the predicted and true labels.

The last method defined is eval_metric. 
This method takes six parameters - embs, model, df, args, device, and an optional threshold value. 
The method calls the eval_model method to get predicted and true labels, and then computes various evaluation metrics such as AUC, precision, recall, and F1 scores using the sklearn.metrics library. 
The method returns these evaluation metrics. The threshold value is used to convert the predicted scores into binary labels, and the specific value of the threshold may depend on the particular dataset being used.
"""

class ScorePredictor(nn.Module):
    def __init__(self, args, **params):
        super().__init__()
        self.args = DotMap(args.toDict())
        for k,v in params.items():
            self.args[k] = v
        
        if self.args.predictor == 'dot':
            pass
        elif self.args.predictor == '1-linear':
            self.predictor = nn.Linear(self.args.dim_embs*2, 1)
        elif self.args.predictor == '2-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1))
        elif self.args.predictor == '3-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1)
                                         )
        elif self.args.predictor == '4-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1)
                                         )
        self.reset_parameters()
            
    def reset_parameters(self):
        pass

    def forward(self, u_e, u_v):
        if self.args.predictor == 'dot':
            score = u_e.mul(u_v).sum(dim=-1)
        else:
            x = torch.cat([u_e, u_v], dim=-1)
            score = self.predictor(x).flatten()
        return score

def eval_model(embs, model, df, batched, args, device):
    if batched:
        dataset = LabelPairs(df)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.loss_batch_size, num_workers=args.num_workers, shuffle=True)
        y_pre_list = []
        y_true_list = []
        for pair, y in dataloader:
            uids, vids = pair.T
            score  = model.predict_combine(embs, uids, vids, device)
            y_pre_list.append(torch.sigmoid(score))
            y_true_list.append(y)
        y_pre = torch.cat(y_pre_list, dim=-1).cpu().numpy()
        y_true = torch.cat(y_true_list, dim=-1).cpu().numpy()
    else:
        uids = torch.from_numpy(df.src.values).long()
        vids = torch.from_numpy(df.dst.values).long()
        score  = model.predict_combine(embs, uids, vids, device)
        y_pre = torch.sigmoid(score).cpu().numpy()
        y_true = df['label'].values
    return y_true, y_pre
    
def eval_metric(embs, model, df, args, device, threshold=0.05):
	# change threshold according to different datasets
	# 0.05 for Alpha, 0.1 for OTC
    y_true, y_pre = eval_model(embs, model, df, args.eval_batched, args, device)
    y = (y_pre > threshold)
    auc = metrics.roc_auc_score(y_true, y_pre)
    prec = metrics.precision_score(y_true, y)
    recl = metrics.recall_score(y_true, y)
    binary_f1 = metrics.f1_score(y_true, y, average='binary')
    micro_f1 = metrics.f1_score(y_true, y, average='micro')
    macro_f1 = metrics.f1_score(y_true, y, average='macro')
    
    
    return auc, prec, recl, micro_f1, binary_f1, macro_f1




############################################### Training Parameter Setting ############################################################

args = DotMap()
args.num_nodes = graph_user.num_nodes()

args.pos_edge_type = ['positive']
args.neg_edge_type = ['negative']
args.edge_type = args.pos_edge_type+args.neg_edge_type
args.num_edge_types = len(args.edge_type)
args.dim_features = graph_user.nodes['user'].data['feature'].shape[1]
args.dim_hiddens = args.dim_features*2
args.dim_embs = args.dim_features*2

args.learning_rate = 0.01

args.conv_depth = 2
args.loss_batch_size = 102400             # to calculated loss

#args.inference_batch_size = 128       # the batch size for inferencing all/batched nodes embeddings
args.sampling_batch_size = 128
args.residual = False
args.num_heads = 8
args.dropout = 0

# active_tag walktogether_tag  friend_tag  playagain_tag 
# label
args.label = 'label'  
args.conv_type = 'gat'
args.het_agg_type = 'attn' # multiplex aggregation
args.dim_query = args.dim_features*2
args.predictor = '2-linear'
# concat / mean / attn / pos
args.combine_type = 'concat'

# sign / common
args.sign_conv = 'sign'
# pos / neg / both
args.sign_aggre = 'both'
# pos / neg / intra / inter / all
args.contrast_type = 'all'
# delete / change / reverse / composite
args.augment = 'change'

#args.contrastive = True
args.mask_ratio = 0.1
args.tao = 0.05
args.alpha = 1e-4
args.beta = 0.8
args.pos_gamma = 1
args.neg_gamma = 1

args.gpu = 0
args.num_workers = 0
args.verbose = 1
args.pretrain_epochs = 101
args.finetune_epochs = 0
# both / pos / neg
args.drop_type = 'both'

# 2-layer 20

device = torch.device(f'cuda:{args.gpu}')




import pickle

import numpy as np

label_train = df_train
label_test = df_test
label_val = df_val
label_ids = np.unique(np.concatenate((label_train.src, label_train.dst, label_test.src, label_test.dst)))


############################################### Training ############################################################
"""
This code is a Python script that trains a graph embedding model and evaluates its performance on a test dataset. 
The script starts by initializing an empty list test_results. 
Then, it enters a loop that will run once, which defines a subgraph of the main graph graph_user based on the input edge type args.edge_type. 
The script then creates a Model instance and an optimizer (Adam) for training the model. 
The script creates two datasets: dataloader_nodes for training the node embeddings and dataloader_labels for training the label embeddings. 
It also initializes a dictionary res to store the results of training.

The script then enters another loop that runs for args.pretrain_epochs+args.finetune_epochs epochs. 
Within this loop, the script applies graph augmentation to the subgraph and creates batches of nodes and labels to train the model. 
It then calculates the loss on the current batch, computes gradients, and updates the model parameters. 
After every 50 epochs, the script evaluates the performance of the model on the training and test datasets and stores the results in the res dictionary.

After completing the training loop, the script empties the cache and calculates the mean of the test results for each evaluation metric. 
It stores the mean values in the test_results list.
"""
test_results = []

for m in range(1):
    g = graph_user.edge_type_subgraph(args.edge_type)
    pos_weight = None
    model = Model(args).to(device)
    opt = torch.optim.Adam(model.parameters())

    dataset = NodeBatch(np.arange(g.num_nodes()))
    dataloader_nodes = torch.utils.data.DataLoader(dataset, batch_size=args.loss_batch_size, shuffle=True)


    dataset = LabelPairs(label_train)
    dataloader_labels = torch.utils.data.DataLoader(dataset, batch_size=int(args.loss_batch_size*len(label_train)/g.num_nodes()), shuffle=True)
    res = defaultdict(list)

    for e in range(args.pretrain_epochs+args.finetune_epochs):
        g_attr, g_stru = GraphAug(g, args)
        cnt = 0

        for nids, (pair, y) in zip(dataloader_nodes, dataloader_labels):
            u, v = pair.T
            y = y.to(device)
            nids = torch.unique(torch.cat((u, v), dim=-1))

            
            embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg = model(g_attr, g_stru, nids, device)
            loss_contrastive = model.compute_contrastive_loss(device, embs_attr_pos[nids], embs_stru_pos[nids], embs_attr_neg[nids], embs_stru_neg[nids])

            y_score = model.predict_combine((embs_attr_pos,embs_stru_pos,embs_attr_neg, embs_stru_neg), u, v, device)
            loss_label = model.compute_label_loss(y_score, y, pos_weight, device)

            loss = args.alpha  * loss_contrastive + loss_label

            print(f'epoch:{e}  {cnt}/{len(dataloader_nodes)}  loss_contrastive:{loss_contrastive}  loss_label:{loss_label}.')

            opt.zero_grad() 
            loss.backward()
            opt.step()
            cnt += 1

            
            
            
            with torch.no_grad():
                embs, (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg) = model.inference(g, g, label_ids, device)
                train_auc, train_prec, train_recl, train_micro_f1, train_binary_f1, train_macro_f1 = eval_metric(embs, model, label_train, args, device)
                test_auc, test_prec, test_recl, test_micro_f1, test_binary_f1, test_macro_f1 = eval_metric(embs, model, label_test, args, device)
                val_auc, val_prec, val_recl, val_micro_f1, val_binary_f1, val_macro_f1 = eval_metric(embs, model, label_val, args, device)
                res['train'].append([train_auc, train_prec, train_recl, train_micro_f1, train_binary_f1, train_macro_f1])
                res['test'].append([test_auc, test_prec, test_recl, test_micro_f1, test_binary_f1, test_macro_f1])
                res['val'].append([val_auc, val_prec, val_recl, val_micro_f1, val_binary_f1, val_macro_f1])
                    



