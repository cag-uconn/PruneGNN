import argparse
import os
import os.path as osp
import random
import copy

import yaml
import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import *
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.utils import scatter
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, LinearDecayTheta, CosineDecayTheta

from models import *
from dataset import *

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


def test_sparsity(model):
    """
    Test sparsity for every involved layer and the overall compression rate
    """
    total_zeros = 0
    total_nonzeros = 0

    for i, (name, W) in enumerate(model.named_parameters()):
        
        if 'bias' in name:
            continue
        W = W.cpu().detach().numpy()
        zeros = np.sum(W == 0)
        total_zeros += zeros
        nonzeros = np.sum(W != 0)
        total_nonzeros += nonzeros
        print("Sparsity at layer {} is {}".format(name, float(zeros) / (float(zeros + nonzeros))))


def train(args, model, optimizer, criterion, data, mask=None):

    model.train() 

    out = model(args, data.x, data.edge_index)  
    optimizer.zero_grad()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    if mask is not None: mask.step()
    else: optimizer.step()

    return float(loss.item())

@torch.no_grad()
def test(args, model, data):
    model.eval()
    out = model(args, data.x, data.edge_index)

    accs = []
    
    if args.dataset_name == "Flickr": 
        pred = out.argmax(dim=-1)
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(f1_score(pred[mask].cpu(), data.y[mask].cpu(), average='micro') if pred[mask].sum() > 0 else 0)

    elif args.dataset_name == "Yelp":
        pred = (torch.sigmoid(out) > 0.5).float()
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='micro'))

    else:
        pred = out.argmax(dim=-1)
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accs



def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]
    
def get_pruned_parameters(file):
    with open(file, 'r') as stream:
        out = yaml.full_load(stream)
        return out['weight_names']


def main():
    
    parser = argparse.ArgumentParser(description='Sparse Training with GNNs')
    parser.add_argument('--training_type', choices=['sparse', 'dense'], default='dense',
                        help='sparse or dense training')
    parser.add_argument('--device', default="cuda", 
                        help='cuda or cpu')
    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--dataset_dir', type=str, default="/PruneGNN/datasets", help='Running directory')
    parser.add_argument('--model_type', choices=['GCN', 'GIN', 'GAT'], default='GAT')
    parser.add_argument('--dataset_name', type=str, default='Cora', help='Dataset name: Cora, Pubmed, CiteSeer, Flickr, Yelp, NELL')
    parser.add_argument('--pruning_type', choices=['irregular', 'column'], default='irregular') 
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--hidden_channels', type=int, default=16, metavar='N',
                        help='hidden channels for GNN (default: 16)')     
    parser.add_argument('--sparsity_rate', type=float, default=0.5, metavar='N',
                        help='sparsity rate for model weights')    
    parser.add_argument('--num_cols', type=int, default=16, metavar='N',
                        help='number of columns to keep for column pruning')     
    parser.add_argument('--running-dir', type=str, default=osp.dirname(osp.realpath(__file__)), help='Running directory')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--report_freq', type=int, default=10, help="Print frequency during training") 

    sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args()


    args.device = torch.device(args.device)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    if args.pruning_type == 'irregular':
        args.sparse_init = 'uniform' 
        args.growth = 'gradient'
        args.death = 'magnitude'

    if args.pruning_type == 'column':
        args.sparse_init = 'row_wise_uniform' 
        args.growth = 'ucb_row_wise'
        args.death = 'row_wise'

    # Parse argument for GPUs
    args.gpus = parse_gpus(args.gpus)

    if args.pruning_type == 'column':
        args.density = float(args.num_cols / args.hidden_channels)
        args.sparsity_rate = 1 - args.density
    else:
        args.density = 1 - args.sparsity_rate


    if args.training_type == 'sparse':
        if args.pruning_type == 'column':
            print("Sparse Training with " + args.dataset_name + ' on ' + args.model_type + ' with ' + str(args.num_cols) + ' columns ' ) 
        else:
            print("Sparse Training with " + args.dataset_name + ' on ' + args.model_type + ' with ' + str(args.sparsity_rate))

    else:
        print("Dense Training with " + args.dataset_name + ' on ' + args.model_type)


    # Load dataset
    dataset, data, num_features, num_classes = get_dataset(args, args.dataset_dir + "/" + args.dataset_name + "/")
    
    

    if args.model_type == 'GCN':
        model = GCN(num_features, args.hidden_channels, num_classes)

        if args.dataset_name == 'Cora' or args.dataset_name == 'Pubmed' or args.dataset_name == 'CiteSeer':
            optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.05)   
        elif args.dataset_name == 'Yelp':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
        elif args.dataset_name == 'NELL':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
       
        names = get_pruned_parameters('config/GCN.yaml')
            
    if args.model_type == 'GIN':
        model = GIN(num_features, args.hidden_channels, num_classes)
        if args.dataset_name == 'Cora' or args.dataset_name == 'Pubmed':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
        elif args.dataset_name == 'Flickr':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        elif args.dataset_name == 'NELL':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)#, weight_decay=5e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

        names = get_pruned_parameters('config/GIN.yaml')
     
        
    if args.model_type == 'GAT':
        model = GAT(num_features, int(args.hidden_channels/8), num_classes)
        if args.dataset_name == "NELL":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif args.dataset_name == "Yelp" or args.dataset_name == "Flickr":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)#, weight_decay=5e-4)
        names = get_pruned_parameters('config/GAT.yaml')

    if args.dataset_name == "Yelp":
        criterion = F.binary_cross_entropy_with_logits
    elif args.dataset_name == "Flickr":
        criterion = F.nll_loss
    else:
        criterion = F.cross_entropy    

    
    print('Using ' + ("GPU " + str(args.gpus[0]) if args.device == torch.device('cuda') else "CPU"))

    model = model.to(args.device)

    
    data = data.to(args.device)

    mask = None
    if args.training_type == 'sparse':
        decay = CosineDecay(args.death_rate, args.epochs*args.multiplier)
        decay_theta = LinearDecayTheta(args.theta, args.factor, args.theta_decay_freq)
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, theta_decay=decay_theta, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, theta=args.theta, epsilon=args.epsilon, args=args)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density, weight_names=names)

    
    print(args)
    print('****************Start training****************')
    model.train()
    best_test_acc = 0
    best_val_acc = 0
    for epoch in range(args.epochs+1):
        train_acc = 0
        val_acc = 0
        test_acc = 0

        loss = train(args, model, optimizer, criterion, data, mask)

        train_acc, val_acc, test_acc = test(args, model, data)    

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if args.dataset_name == "Yelp":
            if epoch > 20 and val_acc < best_val_acc:
                print("Early stopping.")
                break
        
        if(epoch % args.report_freq == 0):
            if args.dataset_name == "Yelp" or args.dataset_name =="Flickr":
                print('In the {}th epoch, the loss is: {:.4f}, training F1 score: {:.4f}, validation F1 score: {:.4f}, test F1 score: {:.4f}, best test F1 score: {:.4f}'.format(\
                epoch, loss, train_acc, val_acc, test_acc, best_test_acc))
            else:
                print('In the {}th epoch, the loss is: {:.4f}, training accuracy: {:.4f}, validation accuracy: {:.4f}, test accuracy: {:.4f}, best test accuracy: {:.4f}'.format(\
                epoch, loss, train_acc, val_acc, test_acc, best_test_acc))

    # if args.dataset_name == "Yelp" or args.dataset_name =="Flickr":
    #     print('Finished training, best F1 score: {:.4f}'.format(best_test_acc))
    # else:
    #     print('Finished training, best test accuracy: {:.4f}'.format(best_test_acc))
        
    # if args.training_type == 'sparse':
    #     layer_fired_weights, total_fired_weights = mask.fired_masks_update()

    # print("\n")
    # if args.training_type == "sparse":
    #     test_sparsity(model)

    print("Accuracy Result: ", best_test_acc)
    print("Sparsity Result: ", args.sparsity_rate)
    print("\n\n\n")

if __name__ == '__main__':
   main()

