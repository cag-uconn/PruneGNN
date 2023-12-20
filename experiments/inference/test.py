import sys

import torch
import prune_gnn

from torch_geometric.datasets import *
from torch_geometric.utils import *

device = torch.device('cuda')

# Load graph data
dataset = Planetoid(root = "/MP/datasets/raw", name = 'Cora')
# dataset = Yelp("/MP/datasets/Yelp")

# graph = Reddit(root = "/MP/datasets/raw")
adj = to_torch_csr_tensor(dataset.data.edge_index)

# Prepare input
column_indices = adj.col_indices().to(device=device)
row_pointer = adj.crow_indices()
# degrees = row_pointer[1:] - row_pointer[:-1]
degrees = torch.ones(column_indices.shape).to(device=device)
num_nodes = row_pointer.shape[0]
row_pointer = row_pointer.to(device=device)

data = dataset[0]

# X = data.x.to_dense().to(device=device)

X = torch.rand(num_nodes, 16).to(device=device)
# # Call MP SPMM Row kernel
output1 = prune_gnn.prune_spmm(X, row_pointer, column_indices, degrees, 1, 0, 0.5)
# output2 = mp_gnn.cusparse_spmm_row(X, row_pointer, column_indices, degrees)

# A = to_torch_csr_tensor(dataset.data.edge_index).to(device=device)
# output = torch.sparse.mm(A, X)

# a = torch.ones(1000,60).to(device=device)
# b = torch.ones(60,3).to(device=device)


# output = torch.matmul(a,b)

# output = mp_gnn.cublas_gemm(a,b)

# Print output
# print(output1)
# print(output2)