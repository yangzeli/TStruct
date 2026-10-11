from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TStruct
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import time

import numpy as np

from sklearn.preprocessing import StandardScaler


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'TStruct': TStruct,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        if flag == 'train':
            if hasattr(data_set, 'data_x'):
                print("Calculating mean and std from training data explicitly...")
                self.train_mean = np.mean(data_set.data_x, axis=0)
                self.train_std = np.std(data_set.data_x, axis=0)
                
            else:
                print("Warning: data_set does not have 'data_x' attribute.")

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        criterion =  nn.HuberLoss(delta=0.1)
        return criterion
    def _select_criterion2(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, phase='val'):
        total_loss = 0
        total_count = 0
        total_mae = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, global_index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, :]
                global_index = global_index.to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'former' in self.args.model:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, global_index, flag ='val')
                else:
                    outputs = self.model(batch_x, flag ='val')

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                true = batch_y.detach().cpu()
                pred = outputs.detach().cpu()

          
                mae = F.l1_loss(pred, true) 
                loss = criterion(pred, true) 

                total_loss += loss.item() * batch_x.shape[0]
                total_mae += mae.item() * batch_x.shape[0] 
                total_count += batch_x.shape[0]
        
        if total_count > 0:
            total_loss = total_loss / total_count
            total_mae = total_mae / total_count 
        else:
            total_loss = 0.0
            total_mae = 0.0
            
        self.model.train()
        return total_loss, total_mae 


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        print("model parameters:", sum(p.numel() for p in self.model.parameters()))
        for name, param in self.model.named_parameters():
            print(f"{name}: {tuple(param.shape)}")


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        mse_criterion = self._select_criterion2()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, global_index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, :]
                global_index =global_index.to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                if 'former' in self.args.model:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, global_index, flag = 'train')
                else:
                    outputs = self.model(batch_x, flag ='train')

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
                if (i + 1) % 100 == 0:
                    print(f"\tloss_auxi: {loss_auxi.item()}")

                loss_huber = criterion(outputs, batch_y)
                if (i + 1) % 100 == 0:
                    print(f"\tloss_huber: {loss_huber.item()}") 

                loss = self.args.huber_lambda * loss_huber + self.args.auxi_lambda * loss_auxi

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                self.model.eval()
                vali_loss, vali_mae = self.vali(vali_data, vali_loader, mse_criterion, phase='val')
                test_loss, test_mae = self.vali(test_data, test_loader, mse_criterion, phase='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                if (epoch+1) % 5 == 0:
                    train_mse, train_mae = self.vali(train_data, train_loader, mse_criterion, phase='train')
                    print("Train_mse: {0:.7f} Train_mae: {1:.7f} Vali MAE: {2:.7f} Test MAE: {3:.7f}".format(
                                train_mse, train_mae, vali_mae, test_mae))

                self.model.train()
                early_stopping(vali_loss, self.model, path)
                
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
       
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model from checkpoint...')
            checkpoint_path = self.args.checkpoints
            
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                print(f"Successfully loaded model from {checkpoint_path}")
            else:
                print(f"Error: Checkpoint not found at {checkpoint_path}")
                return 

        total_mse = 0
        total_mae = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, global_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, :]
                global_index =global_index.to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'former' in self.args.model:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, global_index, flag ='test')
                else:
                    outputs = self.model(batch_x, flag ='test')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                if outputs.ndim == 4:
                    pred = np.transpose(outputs[:, :, :, 1], (0, 2, 1))
                else:
                    pred = outputs
               
                batch_size = pred.shape[0]

                b_mae, b_mse, _, _, _, _, _ = metric(pred, true)
                
                # Accumulation (metric * sample size)
                total_mse += b_mse * batch_size
                total_mae += b_mae * batch_size
                total_samples += batch_size
            
        mse = total_mse / total_samples
        mae = total_mae / total_samples

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")

        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()
        return

  
