import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.CGCNN_data import collate_pool,get_train_val_test_loader,CIFData
from model.CGCNN_model import CrystalGraphConvNet,Normalizer,mae,AverageMeter

class FineTune(object):
    def __init__(self,root_dir,save_dir,unit,tar,log_every_n_steps,eval_every_n_epochs,epoch,opti,lr,momentum,
                 weight_decay,cif_list,batch_size,n_conv,random_seed = 1129,pin_memory=False):
        self.lr = lr
        self.tar = tar
        self.opti = opti
        self.unit = unit
        self.epochs = epoch
        self.data = cif_list
        self.n_conv = n_conv
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.momentum = momentum
        collate_fn = collate_pool
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_epochs = eval_every_n_epochs
        self.criterion = nn.MSELoss()
        self.dataset = CIFData(root_dir=self.root_dir,data_file=self.data,unit=self.unit,tar=self.tar,max_num_nbr=12,radius=8,dmin=0,step=0.2,random_seed=1129)
        self.device = self._get_device()
        self.model_checkpoints_folder = save_dir + "checkpoints/"
        self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(dataset = self.dataset,
                                                                                            random_seed = self.random_seed,
                                                                                            collate_fn = collate_fn,
                                                                                            pin_memory = self.pin_memory,
                                                                                            batch_size = self.batch_size)
        sample_data_list = [self.dataset[i] for i in range(len(self.dataset))]
        _, sample_target, _ = collate_pool(sample_data_list)
        self.normalizer = Normalizer(sample_target)
        with open(save_dir + 'normalizer.pkl', 'wb') as f:
            pickle.dump(self.normalizer, f)

    def _get_device(self):
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.set_device(0)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = self.n_conv,n_out=15)
        if self.device == 'cuda':
            torch.cuda.set_device(0)
            model.to(self.device)
            print("Use cuda for torch")
        else:
            print("Only use cpu for torch")
        layer_list = []
        for name, _ in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        if self.opti == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.lr}, {'params': params}],
                 self.lr, momentum=self.momentum, 
                weight_decay=self.weight_decay)
        elif self.opti == 'Adam':
            lr_multiplier = 0.2
            optimizer = optim.Adam([{'params': base_params, 'lr': self.lr*lr_multiplier}, {'params': params}],
            self.lr, weight_decay=self.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        n_iter = 0
        valid_n_iter = 0
        best_valid_mae = np.inf
        errot_all =[]
        for epoch_counter in range(self.epochs):
            for bn, (input, target, _) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                 input[1],
                                 input[2],
                                 input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = target_normed.to(self.device, non_blocking=True)
                else:
                    target_var = target_normed
                output = model(*input_var)
                loss = self.criterion(output, target_var)
                mae_error=mae(self.normalizer.denorm(output.data.cpu()), target)
                if bn % self.log_every_n_steps == 0:
                    print('Epoch: %d, Batch: %d, Loss '%(epoch_counter+1, bn), loss.item())
                errot_all.append(mae_error)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
            if epoch_counter % self.eval_every_n_epochs == 0:
                _,valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(self.model_checkpoints_folder, 'model.pth'))
                valid_n_iter += 1
            error_all_tensors = torch.stack(errot_all)
            mean_MAE = torch.mean(error_all_tensors).item()
            print('Epoch {} Train: MAE {:.3f}'.format(epoch_counter+1, mean_MAE))
        self.model = model
    
    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()
        error_all = []
        with torch.no_grad():
            model.eval()
            for bn, (input, target, _) in enumerate(valid_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                input[1],
                                input[2],
                                input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = target_normed.to(self.device, non_blocking=True)
                else:
                    target_var = target_normed
                output = model(*input_var)
                loss = self.criterion(output, target_var)
                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                print('Epoch [{0}] Validate: [{1}/{2}],''MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch+1, bn+1, len(self.valid_loader), loss=losses, mae_errors=mae_errors))
                error_all.append(mae_error)
        model.train()
        error_all_tensors = torch.stack(error_all)
        mean_MAE = torch.mean(error_all_tensors).item()
        print('Epoch {} Validate: MAE {:.3f}'.format(n_epoch + 1, mean_MAE))
        return losses.avg, mae_errors.avg

    def test(self):
        model_path = os.path.join(self.model_checkpoints_folder,'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        losses = AverageMeter()
        mae_errors = AverageMeter()
        error_all = []
        with torch.no_grad():
            self.model.eval()
            for bn, (input, target, _) in enumerate(self.test_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                input[1],
                                input[2],
                                input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = target_normed.to(self.device, non_blocking=True)
                else:
                    target_var = target_normed
                output = self.model(*input_var)
                loss = self.criterion(output, target_var)
                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                print('Test: [{0}/{1}], '
                'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))
                error_all.append(mae_error)
        self.model.train()
        error_all_tensors = torch.stack(error_all)
        mean_MAE = torch.mean(error_all_tensors).item()
        print('Test: ''MAE {}'.format(mean_MAE))
        return losses.avg, mae_errors.avg
    
    def predict(self):
        model_path = os.path.join(self.model_checkpoints_folder,'model.pth')
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        with torch.no_grad():
            self.model.eval()
            for _, (input, target, batch_cif_ids) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                input[1],
                                input[2],
                                input[3])
                output = self.model(*input_var)
                output = self.normalizer.denorm(output.data.cpu())
                with open(os.path.join(self.save_dir, 'train.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        line = f"{id}, {t.tolist()}, {o.tolist()}\n"
                        f.write(line)

            for _, (input, target, batch_cif_ids) in enumerate(self.valid_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                input[1],
                                input[2],
                                input[3])
                output = self.model(*input_var)
                output = self.normalizer.denorm(output.data.cpu())
                with open(os.path.join(self.save_dir, 'val.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        line = f"{id}, {t.tolist()}, {o.tolist()}\n"
                        f.write(line)
            for _, (input, target, batch_cif_ids) in enumerate(self.test_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                input[1].to(self.device, non_blocking=True),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0],
                                input[1],
                                input[2],
                                input[3])
                output = self.model(*input_var)
                output = self.normalizer.denorm(output.data.cpu())
                with open(os.path.join(self.save_dir, 'test.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        line = f"{id}, {t.tolist()}, {o.tolist()}\n"
                        f.write(line)
        return "sucess predict"