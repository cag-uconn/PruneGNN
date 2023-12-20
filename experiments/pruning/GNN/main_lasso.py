import argparse
import os
import os.path as osp
import random
import copy
import yaml
import numpy as np

import torch
import torch_geometric
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

# Set a seed value for reproducibility
seed = 42
set_seed(seed)

class Prune():
    def __init__(
        self,
        model,
        initial_state_dict,
        pretrain_step=0,
        sparse_step=0,
        frequency=100,
        prune_dict={},
        restore_sparsity=False,
        fix_sparsity=False,
        balance='none',
        prune_device='default'):
        self._model = model
        self.initial_state_dict = initial_state_dict
        self._t = 0
        self._initial_sparsity = {} # 0
        self._pretrain_step = pretrain_step
        self._sparse_step = sparse_step
        self._frequency = frequency
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._balance = balance
        self._prune_device = prune_device
        self._mask = {}
        self._prepare()

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    if (self._balance == 'fix') and (len(parameter.shape) == 4) and (parameter.shape[1] < 4):
                        self._prune_dict.pop(name)
                        print("The parameter %s cannot be balanced pruned and will be deleted from the prune_dict." % name)
                        continue
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                        self._initial_sparsity[name] = 1 - mask.sum().numpy().tolist() / weight.view(-1).shape[0]
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)

    def _update_mask(self, name, weight, keep_k_row):
        if keep_k_row >= 1:
            norm = weight.abs().sum(axis=1)
        
            thrs = torch.topk(weight.abs().sum(axis=1), keep_k_row)[0][-1]
            rows = (norm >= thrs).long()
        
            mask = rows.unsqueeze(0).expand(weight.shape[1], weight.shape[0]).transpose(0, 1).float()
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = self._pretrain_step < self._t < self._pretrain_step + self._sparse_step
        condition3 = self._t == 0 
        return condition1 and condition2 and condition3

    def _get_weight(self, parameter):
        if self._prune_device == 'default':
            weight = parameter.data
        elif self._prune_device == 'cpu':
            weight = parameter.data.to(device=torch.device('cpu'))
        return weight

    def prune(self, args, hardprune=False):
        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._update_mask_conditions() or hardprune:
                        # print("****************************** Round " + str(int(self._t/self._frequency)) + " ******************************")
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (self._t - self._pretrain_step) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = target_sparsity + (self._initial_sparsity[name] - target_sparsity) * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        keep_k_row = int(weight.view(-1).shape[0] * (1.0 - current_sparsity)/weight.shape[1])
                    
                        self._update_mask(name, weight, args.num_cols)

                    if "weight" in name:
                        weight_dev = parameter.device
                        parameter.data = torch.from_numpy(self._mask[name].cpu().numpy() * self.initial_state_dict[name].cpu().numpy()).to(weight_dev)
                    if "bias" in name:
                        parameter.data = self.initial_state_dict[name]
                  

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


def train(args, model, optimizer, criterion, data, prune, pretrain=False):

    model.train() 
  
    out = model(args, data.x, data.edge_index)  
    optimizer.zero_grad()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()

    if pretrain == False:
        prune.prune(args, False)

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

    parser = argparse.ArgumentParser(description='LASSO regression based pruning with GNNs')
    parser.add_argument('--training_type', choices=['sparse', 'dense'], default='dense',
                        help='sparse or dense training')
    parser.add_argument('--device', default="cuda", 
                        help='cuda or cpu')
    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--dataset_dir', type=str, default="/PruneGNN/datasets", help='Running directory')
    parser.add_argument('--model_type', choices=['GCN', 'GIN', 'GAT'], default='GAT')
    parser.add_argument('--dataset_name', type=str, default='Cora', help='Dataset name: Cora, Pubmed, CiteSeer, Flickr, Yelp, NELL')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--pruning_type', choices=['irregular', 'column'], default='irregular') 
    parser.add_argument('--hidden_channels', type=int, default=16, metavar='N',
                        help='hidden channels for GNN (default: 16)')     
    parser.add_argument('--sparsity_rate', type=float, default=0.5, metavar='N',
                        help='sparsity rate for model weights')    
    parser.add_argument('--num_cols', type=int, default=16, metavar='N',
                        help='number of columns to keep for column pruning')     
    parser.add_argument('--running-dir', type=str, default=osp.dirname(osp.realpath(__file__)), help='Running directory')
    parser.add_argument('--report_freq', type=int, default=10, help="Print frequency during training") 

    args = parser.parse_args()
    args.device = torch.device(args.device)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Parse argument for GPUs
    args.gpus = parse_gpus(args.gpus)

    args.density = float(args.num_cols / args.hidden_channels)
    args.sparsity_rate = 1 - args.density
 
    # Dataset Directory
    dataset_dir = args.running_dir + '/dataset/' + args.dataset_name + '/'


    print("Sparse Training with " + args.dataset_name + ' on ' + args.model_type + ' with ' + str(args.num_cols) + ' columns ')

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
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
        names = get_pruned_parameters('config/GAT.yaml')

        
    num_steps_per_epoch = 1
    prune_dict = {}

    for name in names:
        prune_dict[name] = args.sparsity_rate
        # print(name)
        # print(args.sparsity_rate)

    freq = num_steps_per_epoch*args.epochs 
    

    if args.dataset_name == "Yelp":
        criterion = F.binary_cross_entropy_with_logits
    elif args.dataset_name == "Flickr":
        criterion = F.nll_loss
    else:
        criterion = F.cross_entropy    
    print('Using ' + ("GPU " + str(args.gpus[0]) if args.device == torch.device('cuda') else "CPU"))

    model = model.to(args.device)
    initial_state_dict = copy.deepcopy(model.state_dict())
    data = data.to(args.device)

    print(args)
    print('****************Start training****************')
    model.train()
    best_test_acc = 0
    best_val_acc = 0
    pretrain = True
    prune = None
    best_model = None
    for epoch in range(int(args.epochs/2+1)):
        train_acc = 0
        val_acc = 0
        test_acc = 0

        loss = train(args, model, optimizer, criterion, data, prune, pretrain)
        train_acc, val_acc, test_acc = test(args, model, data)    

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model)
          
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if args.dataset_name == "Yelp":
            if epoch > 20 and val_acc <= best_val_acc:
                print("Early stopping as acc is not improving.")
                break

        if(epoch % args.report_freq == 0):
            if args.dataset_name == "Yelp" or args.dataset_name =="Flickr":
                print('In the {}th epoch, the loss is: {:.4f}, training F1 score: {:.4f}, validation F1 score: {:.4f}, test F1 score: {:.4f}, best test F1 score: {:.4f}'.format(\
                epoch, loss, train_acc, val_acc, test_acc, best_test_acc))
            else:
                print('In the {}th epoch, the loss is: {:.4f}, training accuracy: {:.4f}, validation accuracy: {:.4f}, test accuracy: {:.4f}, best test accuracy: {:.4f}'.format(\
                epoch, loss, train_acc, val_acc, test_acc, best_test_acc))

    model = best_model
    train_acc, val_acc, test_acc = test(args, model, data)    

    # print("Before pruning: ", test_acc)
    pretrain = False
    initial_state_dict2 = copy.deepcopy(model.state_dict())
    prune = Prune(model, initial_state_dict2, num_steps_per_epoch * 0, num_steps_per_epoch * args.epochs, freq, prune_dict, False, False, 'none')
    prune.prune(args, True)
  
    train_acc, val_acc, test_acc = test(args, model, data)    

    # print("After pruning: ", test_acc)

    print('****************Start Retraining****************')
    model.train()
    best_test_acc = 0
    best_val_acc = 0
    for epoch in range(int(args.epochs/2+1)):
        train_acc = 0
        val_acc = 0
        test_acc = 0

        loss = train(args, model, optimizer, criterion, data, prune, pretrain)
        train_acc, val_acc, test_acc = test(args, model, data)    

        if test_acc > best_test_acc:
            best_test_acc = test_acc
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if args.dataset_name == "Yelp":
            if epoch > 20 and val_acc <= best_val_acc:
                print("Early stopping as acc is not improving.")
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
        
    # print("\n")
    # test_sparsity(model)

    print("Accuracy Result: ", best_test_acc)
    print("Sparsity Result: ", args.sparsity_rate)
    print("\n\n\n")

if __name__ == '__main__':
   main()

