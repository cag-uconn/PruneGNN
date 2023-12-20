import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, GATv2Conv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool

import torch_geometric.transforms as T

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             bias= False)
        self.conv2 = GCNConv(hidden_channels, out_channels, 
                             bias=False)

    def forward(self, args, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if args.dataset_name == "Flickr":
            return F.log_softmax(x, dim=1) 
        else:
            return x
    
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(Linear(hidden_channels, out_channels))

    def forward(self, args, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        if args.dataset_name == "Flickr":
            return F.log_softmax(x, dim=1) 
        else:
            return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels*heads, out_channels, heads=1)

    def forward(self, args, x, edge_index):
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv2(x, edge_index)
        if args.dataset_name == "Flickr":
            return F.log_softmax(x, dim=1) 
        else:
            return x
