#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch
import random
import copy

import torch_geometric
from torch_geometric.datasets import *
import torch_geometric.transforms as T
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import NeighborLoader, DataLoader
import yaml
from torch_geometric.utils import scatter

import models
from dataset import *

from tqdm import tqdm

import nvtx

from utils import *

def prune_irr(args, input, sparsity):
    return generate_random_sparse_matrix(input.shape[0], input.shape[1], sparsity).to(args.device)

def set_seed(seed):
    # Set the seed for random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def main():
    parser = argparse.ArgumentParser(description='GNN Inference')
    parser.add_argument('--device', default="cuda", 
                        help='cuda or cpu')
    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--dataset_dir', type=str, default="/PruneGNN/datasets", help='Dataset directory')     
    parser.add_argument('--running_dir', type=str, default=osp.dirname(osp.realpath(__file__)), help='Running directory')
    parser.add_argument('--model_type', choices=['GCN', 'GIN', 'GAT', 'DGNN'], default='GCN')
    parser.add_argument('--dataset_name', type=str, default='Yelp', help='Dataset name: Cora, Pubmed, CiteSeer, Flickr, Yelp, NELL')
    parser.add_argument('--num_features', type=int, default=64, metavar='N',
                        help='feature size if the graph does not come with real features')    
    parser.add_argument('--num_labels', type=int, default=3, metavar='N',
                        help='label size if the graph does not come with real labels')     
    parser.add_argument('--hidden_channels', type=int, default=64, metavar='N',
                        help='hidden channels for GNN (default: 16)')     
    parser.add_argument('--compressed_dim', type=int, default=2, metavar='N',
                        help='compressed dimension')     
    
    parser.add_argument('--sparsity_type', type=str, default='structured', help='Sparsity type: irregular, structured')
    parser.add_argument('--algorithm', type=str, default='ST', help='Pruning algorithm: lasso, ST')
    parser.add_argument('--mode', type=str, default='inference', help='Mode: inference, training')
    parser.add_argument('--kernel_type', type=str, default='pruneSp', help='Kernel type: pruneSp, cusparse')
    parser.add_argument('--sparsity_rate', type=float, default=0.999, metavar='N',
                        help='sparsity rate for model weights')    
    
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--gin_flag', type=int, default=0, metavar='N',
                            help='automatically set to 1 when running gin')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='N',
                            help='epsilon for gin model')
    
    parser.add_argument('--tpw', type=int, default=1, metavar='N',
                            help='threads per warp for GNN')
    
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                            help='learning rate (default: 0.01)')
  
    args = parser.parse_args()
    args.device = torch.device(args.device)

    # Parse argument for GPUs
    args.gpus = parse_gpus(args.gpus)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Dataset Directory
    dataset_dir = args.running_dir + '/dataset/' + args.dataset_name + '/'

    # Load dataset
    dataset = get_dataset(args)
    # num_nodes, num_features, num_edges, num_labels, A, X, correct_labels = get_dataset(args)

    if args.model_type != "DGNN":
        num_nodes, num_features, num_labels, A, X, correct_labels = dataset
        if args.sparsity_type == "structured": 
            dim = args.compressed_dim
        else:
            dim = args.hidden_channels
    else:
        num_nodes, num_features, gru_feat, A, X = dataset

        if args.sparsity_type == "structured": 
            dim = int(args.hidden_channels * (1 - args.sparsity_rate))
            gru_dim = int(gru_feat * (1 - args.sparsity_rate))
        else:
            dim = args.hidden_channels
            gru_dim = gru_feat

  
    if args.model_type == 'GCN':
        model = models.GCN(args, num_features, dim, num_labels).to(device=args.device)
    if args.model_type == 'GIN':
        model = models.GIN(args, num_features, dim, num_labels).to(device=args.device)
        args.gin_flag = 1
    if args.model_type == 'GAT':
        model = models.GAT(args, num_features, dim, num_labels).to(device=args.device)
    if args.model_type == "DGNN":
        model = models.DGNN(args, num_features, dim, gru_dim).to(device=args.device)

    
    if args.sparsity_type == "structured": 
        for i, (name, W) in enumerate(model.named_parameters()):
            W.data = torch.rand(W.shape).to(device=args.device)

    print("Number of nodes: ", num_nodes)
    print("Number of features: ", num_features)
    print("Dim: ", dim)
    if args.model_type == "DGNN":
        print("GRU Dim: ", gru_dim)

    print("Inference...")
    for i in range(args.epochs):
        print(f"Epoch {i+1} / {args.epochs}")
        out = model(X, A, args)

if __name__ == "__main__":
    main()