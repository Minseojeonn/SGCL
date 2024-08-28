import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import random
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import math
import os
from collections import defaultdict
from itertools import product
import random
from dotmap import DotMap
import mlflow
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
from CLS import *
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

def load_data(
    dataset_path: str,
    direction: bool,
    node_idx_type: str
) -> np.array:
    """Read data from a file

    Args:
        dataset_path (str): dataset_path
        direction (bool): True=direct, False=undirect
        node_idx_type (str): "uni" - no intersection with [uid, iid], "bi" - [uid, iid] idx has intersection

    Return:
        array_of_edges (array): np.array of edges
        num_of_nodes: [type1(int), type2(int)]
    """
    edgelist = []
    with open(dataset_path) as f:
        for line in f:
            a, b, s = map(int, line.split('\t'))
            if s == -1:
                s = 0
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)

    if node_idx_type.lower() == "uni":
        for idx, edge in enumerate(edgelist.tolist()):
            fr, to, sign = edge
            edgelist[idx] = (fr, to+num_of_nodes[0], sign)
        edgelist = np.array(edgelist)
        assert len(set(edgelist[:, 0].tolist()).intersection(
            set(edgelist[:, 1].tolist()))) == 0, "something worng"

    if direction == False:
        edgelist = edgelist.tolist()
        for idx, edgelist in enumerate(edgelist):
            fr, to, sign = edgelist
            edgelist.append(to, fr, sign)
        edgelist = np.array(edgelist)

    num_edges = np.array(edgelist).shape[0]

    if node_idx_type.lower() == "bi" and direction == False:
        raise Exception("undirect can not use with bi type.")

    return edgelist, num_of_nodes, num_edges

def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
def split_data(
    array_of_edges: np.array,
    split_ratio: list,
    seed: int,
    dataset_shuffle: bool,
) -> dict:
    """Split your dataset into train, valid, and test

    Args:
        array_of_edges (np.array): array_of_edges
        split_ratio (list): train:test:val = [float, float, float], train+test+val = 1.0 
        seed (int) = seed
        dataset_shuffle (bool) = shuffle dataset when split

    Returns:
        dataset_dict: {train_edges : np.array, train_label : np.array, test_edges: np.array, test_labels: np.array, valid_edges: np.array, valid_labels: np.array}
    """

    assert np.isclose(sum(split_ratio), 1), "train+test+valid != 1"
    train_ratio, valid_ratio, test_ratio = split_ratio
    train_X, test_val_X, train_Y, test_val_Y = train_test_split(
        array_of_edges[:, :2], array_of_edges[:, 2], test_size=1 - train_ratio, random_state=seed, shuffle=dataset_shuffle)
    val_X, test_X, val_Y, test_Y = train_test_split(test_val_X, test_val_Y, test_size=test_ratio/(
        test_ratio + valid_ratio), random_state=seed, shuffle=dataset_shuffle)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def get_num_nodes(
    dataset: np.array
) -> int:
    """get num nodes when bipartite

    Args:
        dataset (np.array): dataset

    Returns:
        num_nodes tuple(int, int): num_nodes_user, num_nodes_item
    """
    num_nodes_user = np.amax(dataset[:, 0]) + 1
    num_nodes_item = np.amax(dataset[:, 1]) + 1
    return (num_nodes_user.item(), num_nodes_item.item())


def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            
def generate_mask(mask_ratio, row, column):
    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0,1,size=(row, column))
    arr_mask = np.ma.masked_array(arr_mask_ratio, mask=(arr_mask_ratio<mask_ratio)).filled(0)
    arr_mask = np.ma.masked_array(arr_mask, mask=(arr_mask>=mask_ratio)).filled(1)
    return arr_mask


def generate_attr_graph(g, args):
    # generate noise g_attr
    feature = g.ndata['feature']
    attr_noise = np.random.normal(loc=0, scale=0.1, size=(feature.shape[0], feature.shape[1]))
    attr_mask = generate_mask(args.mask_ratio, row=feature.shape[0], column=feature.shape[1])
    noise_feature = feature*attr_mask + (1-attr_mask) * attr_noise
    
    g_attr = g
    g_attr.ndata['feature'] = noise_feature.float()
    return g_attr

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
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0])).squeeze()
        
        # delete edges
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)
        
        # add reverse_edges
        g_stru = dgl.add_edges(g_stru, etype_edges[1][drop_eids], etype_edges[0][drop_eids], etype=etype)
    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru

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

def logging_with_mlflow_metric(results):
    # metric selected by valid score
    best_auc_epoch, best_auc_score = -1, -float("inf")
    best_macro_epoch, best_macro_score = -1, -float("inf")
    best_binary_epoch, best_binary_score = -1, -float("inf")
    best_micro_epoch, best_micro_score = -1, -float("inf")
    for idx in range(len(results['train'])):
        train_metric, val_metric, test_metric = results['train'][idx], results['val'][idx], results['test'][idx]
        val_auc, val_mi, val_bi, val_ma = val_metric
    
        if best_auc_score <= val_auc:
            best_auc_score = val_auc
            best_auc_epoch = idx
        if best_binary_score <= val_bi:
            best_binary_epoch = val_bi
            best_binary_epoch = idx
        if best_micro_score <= val_mi:
            best_micro_score = val_mi
            best_micro_epoch = idx
        if best_macro_score <= val_ma:
            best_macro_epoch = val_ma
            best_macro_epoch = idx
    
        metrics_dict = {
            "train_auc": train_metric[0],
            "train_binary_f1": train_metric[2],
            "train_macro_f1": train_metric[3],
            "train_micro_f1": train_metric[1],
            "val_auc": val_metric[0],
            "val_binary_f1": val_metric[2],
            "val_macro_f1": val_metric[3],
            "val_micro_f1": val_metric[1],
            "test_auc": test_metric[0],
            "test_binary_f1": test_metric[2],
            "test_macro_f1": test_metric[3],
            "test_micro_f1": test_metric[1]
        }
        mlflow.log_metrics(metrics_dict, synchronous=False, step=idx)

    best_metrics_dict = {
        "best_auc_val": results["val"][best_auc_epoch][0],
        "best_bi_val": results["val"][best_auc_epoch][2],
        "best_ma_val": results["val"][best_auc_epoch][3],
        "best_mi_val": results["val"][best_auc_epoch][1],
        "best_auc_test": results["test"][best_auc_epoch][0],
        "best_bi_test": results["test"][best_auc_epoch][2],
        "best_ma_test": results["test"][best_auc_epoch][3],
        "best_mi_test": results["test"][best_auc_epoch][1]
    }
    mlflow.log_metrics(best_metrics_dict, synchronous=True)