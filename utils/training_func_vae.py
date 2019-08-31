import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import shutil

import torch
from torch import nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms



def train_epoch_vae(epoch,
                    train_loader,
                    model, 
                    optimizer,
                    loss_fn,
                    device):
    model.train()
    total = 0
    train_loss = 0
    train_kld = 0
    train_mse = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss, kld, mse = loss_fn(output, data)
        loss.backward()
        train_loss += loss.item()
        train_kld += kld.item()
        train_mse += mse.item()
        optimizer.step()
        total += data.size(0)
        
    return {'loss':train_loss/total, 
             'kld': train_kld/total,
             'mse': train_mse/total}
    
    
def val_epoch_vae(epoch,
                  loader,
                  model,
                  loss_fn, 
                  device):
    total = 0
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_kld = 0
        val_mse = 0        
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device).float()
            output = model(data)
            loss, kld, mse = loss_fn(output, data)
            val_loss += loss.item()
            val_kld += kld.item()
            val_mse += mse.item()
            total += data.size(0)
    return {'loss':val_loss/total, 
             'kld': val_kld/total,
             'mse': val_mse/total}
            
def train_vae(model, 
              train_loader, 
              val_loader, 
              epochs, 
              lr = 0.001,
              verbose=0,
              SAVE_PATH=None,
              model_name=None,
              save_freq=5,
              log_interval=10,
              device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):    
    
    def vae_loss(output, target, KLD_weight=1):
        """loss is BCE + KLD. target is original x"""
        recon_x, mu_logvar = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]

        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        MSE = F.mse_loss(recon_x, target, reduction='sum')

        loss = MSE + KLD_weight*KLD
        return loss, KLD, MSE

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(np.ceil(epochs/3)), gamma=0.2)

    metrics = {}    
    metrics['train'] = {}
    metrics['val'] = {}
    metrics['val']['best_loss'] = 1000000
    
    for epoch in (tqdm(range(int(epochs))) if verbose>0 else range(int(epochs))):
        # train for one epoch
        train_metrics = train_epoch_vae(epoch=epoch, 
                                       train_loader=train_loader, 
                                       model=model,
                                       optimizer=optimizer,
                                       loss_fn=vae_loss,
                                       device=device)        
        # evaluate on validation set
        val_metrics  = val_epoch_vae(epoch=epoch, 
                                         loader=val_loader, 
                                         model=model, 
                                         loss_fn=vae_loss,
                                         device=device)
        
        for key, val in train_metrics.items():
            metrics['train'].setdefault(key, []).append(val)
        for key, val in val_metrics.items():
            metrics['val'].setdefault(key, []).append(val)
        scheduler.step()
        is_best = False
        if (val_metrics['loss'] < metrics['val']['best_loss']):
            metrics['val']['best_loss'] = min(val_metrics['loss'], metrics['val']['best_loss'])
            is_best = True
        if SAVE_PATH is not None:
            if (is_best or ((epoch % save_freq)==0)):
                save_checkpoint({
                    'model_name': model_name,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': metrics['val']['best_loss'],
                    'metrics': metrics,
                }, is_best, model_name, SAVE_PATH, epoch)
        if verbose==2:
            print(f'Train loss: {train_metrics["loss"]}, Valid acc: {val_metrics["loss"]}')
    return metrics
            
            
            
def save_checkpoint(state, is_best, model_name, PATH, epoch=None):
    model_folder = Path(f'{PATH}/{model_name}')
    model_folder.mkdir(parents=True, exist_ok=True)
    save_path = model_folder / f'e{epoch}_ckpnt.pth.tar'
    torch.save(state, save_path)
    if is_best:
        best_path = model_folder / 'model_best.pth.tar'
        shutil.copyfile(save_path, best_path)

                 
            



# def train_epoch_vae(train_loader, 
#                 model, 
#                 optimizer, 
#                 pred_loss_fn, 
#                 model_loss_fn,
#                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     model= model.train()
#     total = 0
#     loss_total = 0
#     loss_pred_total = 0
#     loss_model_total = 0
#     correct_total = 0
 
#     for i, (inputs, target) in enumerate(train_loader):
#         inputs, target = inputs.to(device), target.type(torch.LongTensor).to(device)
#         output = model(inputs)
#         loss_pred = pred_loss_fn(output, target)
#         if model_loss_fn is not None:
#             loss_model = model_loss_fn(model)
#             loss_model_total += loss_model.item()
#         else:
#             loss_model = 0
#         loss = loss_pred + loss_model
#         pred = output.max(1)[1]
#         correct_total += pred.eq(target.to(device)).sum().item()
#         loss_pred_total += loss_pred.item()
#         loss_total += loss.item()
#         total += inputs.size(0)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return {'acc':correct_total/total,
#             'loss':loss_total/total,
#             'loss_pred': loss_pred_total/total,
#             'loss_model': loss_model_total/total}


# def validate_epoch_vae(val_loader, 
#                    model, 
#                    pred_loss_fn, 
#                    model_loss_fn, 
#                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
# #     for name, layer in model.layers.items():
# #         print(name, layer.weight.device, layer.bias.device, layer.weight.type(), layer.bias.type())
#     model = model.eval()
#     total = 0
#     loss_total = 0
#     loss_pred_total = 0
#     loss_model_total = 0
#     correct_total = 0
    
#     with torch.no_grad():
#         for i, (inputs, target)  in enumerate(val_loader):
#             inputs, target = inputs.to(device), target.type(torch.LongTensor).to(device)
#             output = model(inputs)
#             loss_pred = pred_loss_fn(output, target)
#             if model_loss_fn is not None:
#                 loss_model = model_loss_fn(model)
#                 loss_model_total += loss_model.item()
#             else:
#                 loss_model = 0
#             loss = loss_pred + loss_model
#             pred = output.max(1)[1]
#             correct_total += pred.eq(target.to(device)).sum().item()
#             loss_pred_total += loss_pred.item()
#             loss_total += loss.item()
#             total += inputs.size(0)

#     return {'acc':correct_total/total,
#             'loss':loss_total/total,
#             'loss_pred': loss_pred_total/total,
#             'loss_model': loss_model_total/total}


# def train_net_vae(model, 
#               train_loader, 
#               val_loader, 
#               epochs, 
#               model_loss_fn=None, 
#               verbose=0, 
#               SAVE_PATH=None,
#               model_name=None,
#               save_freq=5,
#               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     pred_loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=.003)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.2)

#     metrics = {}    
#     metrics['train'] = {}
#     metrics['val'] = {}
#     metrics['val']['best_acc'] = 0
    
#     for epoch in (tqdm(range(int(epochs))) if verbose>0 else range(int(epochs))):
#         # train for one epoch
#         train_metrics = train_epoch(train_loader, 
#                                        model,
#                                        optimizer,
#                                        pred_loss_fn, 
#                                        model_loss_fn,
#                                        device)        
#         # evaluate on validation set
#         val_metrics  = validate_epoch(val_loader, 
#                                          model, 
#                                          pred_loss_fn,
#                                          model_loss_fn,
#                                          device)
        
#         for key, val in train_metrics.items():
#             metrics['train'].setdefault(key, []).append(val)
#         for key, val in val_metrics.items():
#             metrics['val'].setdefault(key, []).append(val)
#         scheduler.step()
#         is_best = False
#         if (val_metrics['acc'] > metrics['val']['best_acc']):
#             metrics['val']['best_acc'] = max(val_metrics['acc'], metrics['val']['best_acc'])
#             is_best = True
#         if SAVE_PATH is not None:
#             if (is_best or ((epoch % save_freq)==0)):
#                 save_checkpoint({
#                     'model_name': model_name,
#                     'state_dict': model.state_dict(),
#                     'optimizer' : optimizer.state_dict(),
#                     'epoch': epoch + 1,
#                     'best_val_acc': metrics['val']['best_acc'],
#                     'metrics': metrics,
#                 }, is_best, model_name, SAVE_PATH, epoch)
#         if verbose==2:
#             print(f'Train acc: {train_metrics["acc"]}, Valid acc: {val_metrics["acc"]}')
#     return metrics
