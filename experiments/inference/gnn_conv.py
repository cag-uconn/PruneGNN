import math
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import prune_gnn
import nvtx
import torch.nn as nn
import torch.sparse as sp
from utils import *
import dgl.sparse

class function_GCN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, A, args, layer_name):
        # ctx.save_for_backward(X, weight)
        # ctx.A = A
        # ctx.args = args
        # ctx.layer_name = layer_name
        
        A_csr, A_dgl, row_pointer, column_indices, degrees = A

        if args.sparsity_type == "irregular":
            
            ### comb = X * W ###
            with nvtx.annotate(message="comb"):
                try:
                    comb = torch.matmul(X, weight) #SpMM or SpGEMM
                except:
                    comb = torch.matmul(X, weight.to_dense()) # sparsity is not high enough for a sparse kernel

            comb_sp = prune_irr(args, comb, args.sparsity_rate)
            ### agg = A * comb ###``
            with nvtx.annotate(message="agg"): 
                # inject sparsity on the fly
                try:
                    agg = torch.matmul(A_csr, comb_sp).to_dense() #SpGEMM
                except:
                    agg = torch.matmul(A_csr, comb) # sparsity is not high enough for a sparse kernel

        if args.sparsity_type == "structured":
            ### comb = X * W ###
            with nvtx.annotate(message="comb"):
                if X.is_sparse_csr:
                    comb = torch.matmul(X, weight)
                else:
                    comb = prune_gnn.cublas_gemm(X, weight)

            ### agg = A * comb ###``
            with nvtx.annotate(message="agg"):
            
                if args.kernel_type == "pruneSp":
                    agg = prune_gnn.prune_spmm(comb, row_pointer, column_indices, degrees, args.tpw, args.gin_flag, args.epsilon)
                elif args.kernel_type == "cusparse":
                    agg = prune_gnn.cusparse_spmm_row(comb, row_pointer, column_indices, degrees)

        return agg

class function_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, A, args, layer_name):
        # print(X.shape)
        # ctx.A = A
        # ctx.args = args
        # ctx.layer_name = layer_name
        
        A_csr, A_dgl, row_pointer, column_indices, degrees = A
        # X = X.to_dense()
       
        if args.sparsity_type == "irregular":

            with nvtx.annotate(message="agg"):

                if X.is_sparse_csr:
                    agg = torch.matmul(A_csr, X).to_dense() #SPGEMM
                    agg = agg + (1 + args.epsilon) * X
                else:
                    agg = prune_gnn.cusparse_spmm_row(X, row_pointer, column_indices, degrees)
                    agg = agg + (1 + args.epsilon) * X

            
            # inject sparsity on the fly
            agg_sp = prune_irr(args, agg, args.sparsity_rate)
                
          
            with nvtx.annotate(message="comb"):
                try:
                    comb = torch.matmul(agg_sp, weight).to_dense() #SpGEMM
                except:
                    comb = torch.matmul(agg, weight) #sparsity is not high enough for a sparse kernel


        if args.sparsity_type == "structured":        
          
            with nvtx.annotate(message="agg"):
                if args.kernel_type == "pruneSp":
                    agg = prune_gnn.prune_spmm(X, row_pointer, column_indices, degrees, args.tpw, args.gin_flag, args.epsilon)
                elif args.kernel_type == "cusparse":
                    # # agg = torch.mm(A_csr, X)
                    agg = prune_gnn.cusparse_spmm_row(X, row_pointer, column_indices, degrees)
                    agg = agg + (1 + args.epsilon) * X

            ctx.save_for_backward(agg, weight)

          
            with nvtx.annotate(message="comb"):
                comb = prune_gnn.cublas_gemm(agg, weight)
      
        return comb
    
class function_GAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_r, weight_l, A, args, layer_name):
        # ctx.save_for_backward(X, weight)
        # ctx.A = A
        # ctx.args = args
        # ctx.layer_name = layer_name
        
        A_csr, A_dgl,  row_pointer, column_indices, degrees = A
        leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU


        if args.sparsity_type == "irregular": 

            with nvtx.annotate(message="comb"):
                try: 
                    comb = torch.matmul(X, weight)     

                except: #sparsity is not high enough for a sparse kernel
                    comb = torch.matmul(X, weight.to_dense())
        
    
            # inject sparsity on the fly
            comb_sp = prune_irr(args, comb, args.sparsity_rate)

            with nvtx.annotate(message="agg"):

                h_r = torch.matmul(X, weight_r).to_dense() #SpMM or SpGEMM
                h_l = torch.matmul(X, weight_l).to_dense() #SpMM or SpGEMM
            
                a_input = dgl.sparse.sddmm(A_dgl, h_l, torch.transpose(h_r, 0, 1))

                A_csr.values = leakyrelu(a_input.val) #update values
                
                try: 
                    agg = torch.matmul(A_csr, comb_sp).to_dense()
                except:
                    agg = torch.matmul(A_csr, comb) #sparsity is not high enough for a sparse kernel

              

            
        if args.sparsity_type == "structured":
           
            with nvtx.annotate(message="comb"):
                if X.is_sparse_csr:
                    comb = torch.matmul(X, weight)
                else:
                    comb = prune_gnn.cublas_gemm(X, weight)

            with nvtx.annotate(message="agg"):
                if X.is_sparse_csr:
                    h_r = torch.matmul(X, weight_r)
                    h_l = torch.matmul(X, weight_l)
                else:
                    h_r = prune_gnn.cublas_gemm(X, weight_r)
                    h_l = prune_gnn.cublas_gemm(X, weight_l)
                
                a_input = dgl.sparse.sddmm(A_dgl, h_l, torch.transpose(h_r, 0, 1))

                degrees = leakyrelu(a_input.val)#update values
                degrees = a_input.val
            
                if args.kernel_type == "pruneSp":
                    agg = prune_gnn.prune_spmm(comb, row_pointer, column_indices, degrees, args.tpw, args.gin_flag, args.epsilon)
                elif args.kernel_type == "cusparse":
                    agg = prune_gnn.cusparse_spmm_row(comb, row_pointer, column_indices, degrees)
              
        return agg


class GCNConv(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim, layer_name):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        if args.sparsity_type == "irregular":
            self.weights_sp = prune_irr(args, self.weights, args.sparsity_rate).to(device=args.device)
        self.reset_parameters()
        self.layer_name = layer_name

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, A, args):
        if args.sparsity_type == "irregular":
            return function_GCN.apply(X, self.weights_sp, A, args, self.layer_name)
        else:
            return function_GCN.apply(X, self.weights, A, args, self.layer_name)


class GINConv(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim, layer_name):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        if args.sparsity_type == "irregular":
            self.weights_sp = prune_irr(args, self.weights, args.sparsity_rate).to(device=args.device)
        self.reset_parameters()
        self.layer_name = layer_name

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, A, args):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        if args.sparsity_type == "irregular":
            return function_GIN.apply(X, self.weights_sp, A, args, self.layer_name)
        else:
            return function_GIN.apply(X, self.weights, A, args, self.layer_name)


class GATConv(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim, heads, layer_name):
        super(GATConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.weights_r = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.weights_l = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        if args.sparsity_type == "irregular":
            self.weights_sp = prune_irr(args, self.weights, args.sparsity_rate).to(device=args.device)
            self.weights_r_sp = prune_irr(args, self.weights_r, args.sparsity_rate).to(device=args.device)
            self.weights_l_sp = prune_irr(args, self.weights_l, args.sparsity_rate).to(device=args.device)
        
        self.reset_parameters()
        self.layer_name = layer_name

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.weights_r.data.uniform_(-stdv, stdv)
        self.weights_l.data.uniform_(-stdv, stdv)


    def forward(self, X, A, args):
        if args.sparsity_type == "irregular":
            return function_GAT.apply(X, self.weights_sp, self.weights_r_sp, self.weights_l_sp, A, args, self.layer_name)
        else:
            return function_GAT.apply(X, self.weights, self.weights_r, self.weights_l, A, args, self.layer_name)
            
class GRUCell(nn.Module):
    def __init__(self, args, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih_up = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_hh_up  = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.weight_ih_res = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_hh_res  = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.weight_ih_new = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_hh_new  = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        if args.sparsity_type == "irregular":
            self.weight_ih_up_sp = prune_irr(args, self.weight_ih_up, args.sparsity_rate).to(device=args.device)
            self.weight_hh_up_sp  = prune_irr(args, self.weight_hh_up, args.sparsity_rate).to(device=args.device)

            self.weight_ih_res_sp = prune_irr(args, self.weight_ih_res, args.sparsity_rate).to(device=args.device)
            self.weight_hh_res_sp  = prune_irr(args, self.weight_hh_res, args.sparsity_rate).to(device=args.device)

            self.weight_ih_new_sp = prune_irr(args, self.weight_ih_new, args.sparsity_rate).to(device=args.device)
            self.weight_hh_new_sp  = prune_irr(args, self.weight_hh_new, args.sparsity_rate).to(device=args.device)

        self.state = None


    def forward(self, args, input):

        if self.state == None:
            self.state = torch.zeros(input.shape[0],self.hidden_size).to(input.device)

        hx = self.state

        if args.sparsity_type == "irregular":

            updategate = torch.mm(input, self.weight_ih_up_sp) 
            updategate = updategate + torch.mm(hx, self.weight_hh_up_sp)
        
            resetgate = torch.mm(input, self.weight_ih_res_sp) 
            resetgate = resetgate + torch.mm(hx, self.weight_hh_res_sp) 
            
            newcan = torch.mm(input, self.weight_ih_new_sp) 
            newcan = newcan + (torch.mm(hx, self.weight_hh_new_sp) * resetgate)
        
        
        else:
            
            updategate = prune_gnn.cublas_gemm(input, self.weight_ih_up) 
            updategate = updategate + prune_gnn.cublas_gemm(hx, self.weight_hh_up)
        
            resetgate = prune_gnn.cublas_gemm(input, self.weight_ih_res) 
            resetgate = resetgate + prune_gnn.cublas_gemm(hx, self.weight_hh_res) 
            
            newcan = prune_gnn.cublas_gemm(input, self.weight_ih_new) 
            newcan = newcan + (prune_gnn.cublas_gemm(hx, self.weight_hh_new) * resetgate)
        
        cy = (updategate * hx) + (1 - updategate) * newcan

        self.state = hx

        return hx
    