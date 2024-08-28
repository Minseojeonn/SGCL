import math
import os
from collections import defaultdict
from itertools import product
import random
from dotmap import DotMap
from utils import logging_with_mlflow_metric
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
from parser import parsing
import mlflow
warnings.filterwarnings("ignore")
def main():
    arg = parsing()
    seed = arg.seed
    dataset_name = arg.dataset_name
    input_dim = arg.input_dim
    device = arg.device
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

    train_positive = train_X[train_Y==1].T
    train_negative = train_X[train_Y==0].T
    tmp_het_data_dict = {('user', 'positive', 'user'): (train_positive[0], train_positive[1]), ('user', 'negative', 'user'): (train_negative[0], train_negative[1])}
    het_data_dict.update(tmp_het_data_dict)


    graph_user = dgl.heterograph(
        data_dict = het_data_dict,
        num_nodes_dict = het_num_nodes_dict
    )
    for node_t in het_node_feat_dict:
        graph_user.nodes[node_t].data['feature'] = torch.from_numpy(het_node_feat_dict[node_t]).float()


    ############################################### Load Labels ############################################################




    """Torch modules for graph convolutions(GCN)."""
    # pylint: disable= no-member, arguments-differ, invalid-name





    args = DotMap()
    args.num_nodes = graph_user.num_nodes()
    args.pos_edge_type = ['positive']
    args.neg_edge_type = ['negative']
    args.edge_type = args.pos_edge_type+args.neg_edge_type
    args.num_edge_types = len(args.edge_type)
    args.dim_features = graph_user.nodes['user'].data['feature'].shape[1]
    args.dim_hiddens = args.dim_features*2
    args.dim_embs = args.dim_features*2

    args.learning_rate = arg.lr

    args.conv_depth = arg.num_layers
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
    args.augment = arg.augment

    #args.contrastive = True
    args.mask_ratio = arg.mask_ratio
    args.tao = arg.tao  
    args.alpha = arg.alpha
    args.beta = arg.beta
    args.pos_gamma = 1
    args.neg_gamma = 1
    args.num_workers = 0
    args.verbose = 1
    args.pretrain_epochs = 101
    args.finetune_epochs = 0
    # both / pos / neg
    args.drop_type = 'both'

    # 2-layer 20

    device = arg.device


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
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
                    res['train'].append([train_auc,train_micro_f1, train_binary_f1, train_macro_f1])
                    res['test'].append([test_auc, test_micro_f1, test_binary_f1, test_macro_f1])
                    res['val'].append([val_auc, val_micro_f1, val_binary_f1, val_macro_f1])
                        
    
    remote_server_uri = "http://-:15001"
    mlflow.set_tracking_uri(remote_server_uri)
    experiment_name = f"Aug19-SGCL-{arg.dataset_name}-{arg.seed}"
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(args)
    logging_with_mlflow_metric(res)
    




main()