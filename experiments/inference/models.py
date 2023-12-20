import math
import torch
import torch.nn.functional as F
from gnn_conv import *
from torch.nn.parameter import Parameter

import nvtx

class GCN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(args, in_channels, hidden_channels, "conv1")
        self.conv2 = GCNConv(args, hidden_channels, out_channels, "conv2")

    @nvtx.annotate(message="forward")
    def forward(self,x, A, args):
        with nvtx.annotate(message="conv1"):
            x = self.conv1(x, A, args)
            # with nvtx.annotate(message="activation"):
            #     x = F.relu(x)

        with nvtx.annotate(message="conv2"):
            x = self.conv2(x, A, args)
        
        return x    
    

class GIN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__()
        # self.in_channels = args.hidden_channels
        # in_channels = args.num_features #self.in_channels 
        self.conv1 = GINConv(args, in_channels, hidden_channels, 'conv1')
        self.conv2 = GINConv(args, hidden_channels, hidden_channels, 'conv2')
        self.conv3 = GINConv(args, hidden_channels, out_channels, 'conv3')
        

    @nvtx.annotate(message="forward")
    def forward(self, x, A, args):

        with nvtx.annotate(message="conv1"):
            x = self.conv1(x, A, args)

        with nvtx.annotate(message="conv2"):
            x = self.conv2(x, A, args)
        
        with nvtx.annotate(message="conv3"):
            x = self.conv3(x, A, args)

        return x

class GAT(torch.nn.Module):
    def __init__(self, args, in_channels, dim, out_channels, heads=1): #note: dim = hidden_channels * heads
        super().__init__()
      
        self.conv1 = GATConv(args, in_channels, dim, heads, "conv1")
        self.conv2 = GATConv(args, dim, out_channels, 1, "conv2")
        

    @nvtx.annotate(message="forward")
    def forward(self, x, A, args):
        # x.to_dense()[:, :self.in_channels].contiguous()
        with nvtx.annotate(message="conv1"):
            x = self.conv1(x, A, args)

        with nvtx.annotate(message="conv2"):
            x = self.conv2(x, A, args)
        
        return x

class DGNN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, gru_feats):
        super().__init__()
        self.conv1 = GCNConv(args, in_channels, hidden_channels, "conv1")
        self.conv2 = GCNConv(args, hidden_channels, hidden_channels, "conv2")
    
        self.rnn = GRUCell(args, hidden_channels, gru_feats)

    @nvtx.annotate(message="forward")
    def forward(self,x, A, args):
        with nvtx.annotate(message="conv1"):
            x = self.conv1(x, A, args)

            
        with nvtx.annotate(message="conv2"):
            x = self.conv2(x, A, args)

      
        with nvtx.annotate(message="GRU"):  
            x = self.rnn(args, x)

        return x    
    