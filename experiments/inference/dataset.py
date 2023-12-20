from torch_geometric.datasets import *
from torch_geometric.utils import *
import torch_geometric.transforms as T
import numpy as np
import torch
import dgl

def get_dataset(args):
    if args.model_type != "DGNN":
        dir = args.dataset_dir + "/raw/" + args.dataset_name
        if args.dataset_name == 'Cora' or args.dataset_name == 'Pubmed' or args.dataset_name == 'CiteSeer':
            dataset = Planetoid(dir, args.dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.x = data.x.to_dense()

        if args.dataset_name == 'NELL':
            dataset = NELL(dir)
           
        if args.dataset_name == 'Flickr':
            dataset = Flickr(dir, transform=T.NormalizeFeatures())
  
        if args.dataset_name == 'Yelp':
            dataset = Yelp(dir)
            
        data = dataset[0]

        if args.dataset_name == 'NELL':   
            num_features = 5414
            data.x = data.x.to_dense()[:, :num_features]
            # convert back to sparse
            data.x = data.x.to_sparse_csr()

        else:
            num_features = dataset.num_features
            

        num_nodes = data.num_nodes
        num_labels = dataset.num_classes

        A = to_torch_csr_tensor(dataset.data.edge_index)

        column_indices = A.col_indices()
        row_pointer = A.crow_indices()
        degrees = A.values()

        X = data.x
        correct_labels = torch.ones(num_nodes).long()
        correct_labels = correct_labels.to(device=args.device)

    else:
        A_dir = args.dataset_dir + "/DGNN/" + args.dataset_name + "/A.pt"
        X_dir = args.dataset_dir + "/DGNN/" + args.dataset_name + "/X.pt"
        A = torch.load(A_dir, map_location=args.device)
        X = torch.load(X_dir)

        if args.dataset_name == "elliptic_temporal" or args.dataset_name == "reddit":
            X = X.to_dense()
        
        if X.is_sparse:
            X = X.to_sparse_csr()
        print(A.shape)
        print(X.shape)

        column_indices = A.col_indices()
        row_pointer = A.crow_indices()
        degrees = A.values()

        num_nodes = A.shape[0]
        num_features = X.shape[1]

        gru_feat = 30


    X = X.to(device=args.device)
    row_pointer = row_pointer.to(device=args.device)
    column_indices = column_indices.to(device=args.device)
    degrees = degrees.to(device=args.device)
    

    A_csr = torch.sparse_csr_tensor(row_pointer.to(dtype=torch.int64),
                            column_indices.to(dtype=torch.int64),
                            degrees.to(dtype=torch.float32), size=(num_nodes,num_nodes))
    
    A_dgl = dgl.sparse.from_csr(A_csr.crow_indices(), A_csr.col_indices(), A_csr.values(), shape=A_csr.shape)
    A = (A_csr, A_dgl, row_pointer, column_indices, degrees)

    if args.model_type == "GIN":
        X = X.to_dense().contiguous()
        # num_features = args.num_features 
        # X = X.to_dense()[:, :num_features].contiguous()
        # #   X = X.to_dense().contiguous()

    if args.model_type != "DGNN":
        dataset = (num_nodes, num_features, num_labels, A, X, correct_labels)
    else:
        dataset = (num_nodes, num_features, gru_feat, A, X)

    return dataset
