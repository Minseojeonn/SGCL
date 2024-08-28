import torch 
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
import math
from utils import *
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
from torch import nn
from torch.nn import init
from torch.functional import F
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils
import numpy as np
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

        dataloader = dgl.dataloading.DataLoader(g, nids,
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_inference),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        y, attn_res = self.calc_from_loader(dataloader, x, device)
        return y, attn_res
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

