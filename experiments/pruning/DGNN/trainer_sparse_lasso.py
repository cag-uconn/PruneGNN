import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import yaml
import copy
import os

def get_pruned_parameters(file):
    with open(file, 'r') as stream:
        out = yaml.safe_load(stream)
        return out['weight_names']


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
        # print('mask updated!!!!!')
        if keep_k_row >= 1:
            # weight = weight.detach().cpu().numpy()
            norm = weight.abs().sum(axis=1)
            # print("norm: ", len(norm), norm)
            
            thrs = torch.topk(weight.abs().sum(axis=1), keep_k_row)[0][-1]
            rows = (norm >= thrs).long()
            # t_rows=torch.ones(2, weight.shape[1])
            # print(rows)
            # print(t_rows)
            mask = rows.unsqueeze(0).expand(weight.shape[1], weight.shape[0]).transpose(0, 1).float()
            # mask = torch.where(weight.abs() >= thrs, torch.ones_like(weight), torch.zeros_like(weight))
            # print("mask: ", mask.size(), mask)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = self._pretrain_step < self._t < self._pretrain_step + self._sparse_step
        # condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        condition3 = self._t == 0 # amit: shaoyi suggested
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
                    # print(weight)
                    if self._update_mask_conditions() or hardprune:
                        # print("****************************** Round " + str(int(self._t/self._frequency)) + " ******************************")
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (self._t - self._pretrain_step) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = target_sparsity + (self._initial_sparsity[name] - target_sparsity) * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        #keep_k_row = int(weight.view(-1).shape[0] * (1.0 - current_sparsity)/weight.shape[1])
                        keep_k_row = int((1-args.sparsity_rate) * weight.shape[0])
                        # print(keep_k_row)
                        # print("current_sparsity: ", current_sparsity)
                        # print("weight.view(-1).shape[0]): ", weight.view(-1).shape[0])
                        # print("keep_k_row: ", keep_k_row)
                        # self._update_mask(name, weight, keep_k_row)
                        self._update_mask(name, weight, keep_k_row)
                        
                        # print((1-args.sparsity_rate) * weight.shape[1])
                        
                        

                    if "weight" in name or "w" in name:
                        weight_dev = parameter.device
                        parameter.data = torch.from_numpy(self._mask[name].cpu().numpy() * self.initial_state_dict[name].cpu().numpy()).to(weight_dev)
                    if "bias" in name:
                        parameter.data = self.initial_state_dict[name]
                    # parameter.mul_(self._mask[name])

                    # print(weight)
class Trainer():
    def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier
        self.comp_loss = comp_loss

        self.num_nodes = dataset.num_nodes
        self.data = dataset
        self.num_classes = num_classes

        self.logger = logger.Logger(args, self.num_classes)
        self.args = args

        self.init_optimizers(args)

        # for name, param in self.gcn.named_parameters():
        #     print(name, param.shape)

        self.names = get_pruned_parameters(args.config_file)

        # decay = CosineDecay(args.death_rate, args.num_epochs*args.multiplier)
        # decay_theta = LinearDecayTheta(args.theta, args.factor, args.theta_decay_freq)

        # if args.training_type == "sparse":
        # 	self.mask = Masking(self.gcn_opt, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, theta_decay=decay_theta, growth_mode=args.growth,
        # 					redistribution_mode=args.redistribution, theta=args.theta, epsilon=args.epsilon, args=args)
        # 	self.mask.add_module(self.gcn, sparse_init=args.sparse_init, density=args.density, weight_names=self.names)

        self.prune_dict = {}
        self.num_steps_per_epoch = 1
        self.freq = self.num_steps_per_epoch*args.num_epochs 

        self.initial_state_dict = copy.deepcopy(self.gcn.state_dict())
        self.pretrain  = True


        for name in self.names:
            self.prune_dict[name] = args.sparsity_rate
            # print(name)
            # print(args.sparsity_rate)
        
        if self.tasker.is_static:
            adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
            self.hist_adj_list = [adj_matrix]
            self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]


    def init_optimizers(self,args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()


    def optim_step(self,loss):
        self.tr_step += 1

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            
            self.gcn_opt.step()
            self.classifier_opt.step()

        if self.pretrain == False:
            self.prune.prune(self.args, False)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self, filename, model):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
            return 0
        
    def test_sparsity(self,model):
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
            # print("zeros: ", zeros)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            # print("nonzeros: ", nonzeros)
            total_nonzeros += nonzeros
            print("Sparsity at layer {} is {}".format(name, float(zeros) / (float(zeros + nonzeros))))

    def save_weights(self):
        
        for i, (name, W) in enumerate(self.gcn.named_parameters()):
            if 'bias' in name:
                continue
            if name in self.names:
                torch.save(W, self.args.weight_dir + name + '_' + self.args.pruning_type + '_'+ str(self.args.sparsity_rate) + '.pt')
            
         

    def train(self):

        model_path =  self.args.data + ".ckpt"
        if not os.path.exists(model_path):

            self.tr_step = 0
            best_eval_valid = 0
            eval_valid = 0
            epochs_without_impr = 0
            best_model = None

            print("Pre-training")
            for e in range(int(self.args.num_epochs/2)):
                eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
                if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs and (e+1)%self.args.valid_freq==0:
                    eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
                    if eval_valid>best_eval_valid:
                        best_model = copy.deepcopy(self.gcn)
                        best_eval_valid = eval_valid
                        epochs_without_impr = 0
                        # print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
                    else:
                        epochs_without_impr+=1
                        if epochs_without_impr>self.args.early_stop_patience:
                            print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
                            break

                # if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs and (e+1)%self.args.valid_freq==0:
                #     eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)
            
            torch.save(self.gcn.state_dict(), model_path)

            if best_model != None:
                self.gcn = copy.deepcopy(best_model)
        else:
            self.gcn.load_state_dict(torch.load(model_path))

        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        self.pretrain = False
        initial_state_dict2 = copy.deepcopy(self.gcn.state_dict())
        self.prune = Prune(self.gcn, initial_state_dict2, self.num_steps_per_epoch * 0, self.num_steps_per_epoch * self.args.num_epochs, self.freq, self.prune_dict, False, False, 'none')
        self.prune.prune(self.args, True)
        # self.test_sparsity(self.gcn)
        print()

        print("Retraining")
        for e in range(int(self.args.num_epochs/2)):
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
            if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
                if eval_valid>best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
                else:
                    epochs_without_impr+=1
                    if epochs_without_impr>self.args.early_stop_patience:
                        print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
                        break

            # if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
            #     eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

        
        # self.test_sparsity(self.gcn)
        print("Accuracy Result: ", best_eval_valid)
        print("Sparsity Result: ", self.args.sparsity_rate)
        print()
        print()


    def run_epoch(self, split, epoch, set_name, grad):
        t0 = time.time()
        log_interval=999
        if set_name=='TEST':
            log_interval=1
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        ctr = 0 
        for s in split:
            ctr = ctr + 1
            if self.tasker.is_static:
                s = self.prepare_static_sample(s)
            else:
                s = self.prepare_sample(s)

            predictions, nodes_embs = self.predict(s.hist_adj_list,
                                                   s.hist_ndFeats_list,
                                                   s.label_sp['idx'],
                                                   s.node_mask_list)

            loss = self.comp_loss(predictions,s.label_sp['vals'])
            # print(loss)
            if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
            else:
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
                  
        
            if grad:
                self.optim_step(loss)

        torch.set_grad_enabled(True)
        eval_measure = self.logger.log_epoch_done()

        return eval_measure, nodes_embs

    def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
        nodes_embs = self.gcn(hist_adj_list,
                              hist_ndFeats_list,
                              mask_list)

        predict_batch_size = 100000
        gather_predictions=[]
        for i in range(1 +(node_indices.size(1)//predict_batch_size)):
            cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
            predictions = self.classifier(cls_input)
            gather_predictions.append(predictions)
        gather_predictions=torch.cat(gather_predictions, dim=0)
        return gather_predictions, nodes_embs

    def gather_node_embs(self,nodes_embs,node_indices):
        cls_input = []

        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input,dim = 1)




    def prepare_sample(self,sample):
        sample = u.Namespace(sample)
        for i,adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
        else:
            label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def prepare_static_sample(self,sample):
        sample = u.Namespace(sample)

        sample.hist_adj_list = self.hist_adj_list

        sample.hist_ndFeats_list = self.hist_ndFeats_list

        label_sp = {}
        label_sp['idx'] =  [sample.idx]
        label_sp['vals'] = sample.label
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self,adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

        pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
        #print ('Node embs saved in',file_name)