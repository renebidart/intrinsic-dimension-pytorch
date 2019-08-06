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
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms


def train_epoch(train_loader, 
                model, 
                optimizer, 
                pred_loss_fn, 
                model_loss_fn,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model= model.train()
    total = 0
    loss_total = 0
    loss_pred_total = 0
    loss_model_total = 0
    correct_total = 0
 
    for i, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.type(torch.LongTensor).to(device)
        output = model(inputs)
        loss_pred = pred_loss_fn(output, target)
        if model_loss_fn is not None:
            loss_model = model_loss_fn(model)
            loss_model_total += loss_model.item()
        else:
            loss_model = 0
        loss = loss_pred + loss_model
        pred = output.max(1)[1]
        correct_total += pred.eq(target.to(device)).sum().item()
        loss_pred_total += loss_pred.item()
        loss_total += loss.item()
        total += inputs.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {'acc':correct_total/total,
            'loss':loss_total/total,
            'loss_pred': loss_pred_total/total,
            'loss_model': loss_model_total/total}


def validate_epoch(val_loader, 
                   model, 
                   pred_loss_fn, 
                   model_loss_fn, 
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
#     for name, layer in model.layers.items():
#         print(name, layer.weight.device, layer.bias.device, layer.weight.type(), layer.bias.type())
    model = model.eval()
    total = 0
    loss_total = 0
    loss_pred_total = 0
    loss_model_total = 0
    correct_total = 0
    
    with torch.no_grad():
        for i, (inputs, target)  in enumerate(val_loader):
            inputs, target = inputs.to(device), target.type(torch.LongTensor).to(device)
            output = model(inputs)
            loss_pred = pred_loss_fn(output, target)
            if model_loss_fn is not None:
                loss_model = model_loss_fn(model)
                loss_model_total += loss_model.item()
            else:
                loss_model = 0
            loss = loss_pred + loss_model
            pred = output.max(1)[1]
            correct_total += pred.eq(target.to(device)).sum().item()
            loss_pred_total += loss_pred.item()
            loss_total += loss.item()
            total += inputs.size(0)

    return {'acc':correct_total/total,
            'loss':loss_total/total,
            'loss_pred': loss_pred_total/total,
            'loss_model': loss_model_total/total}


def train_net(model, 
              train_loader, 
              val_loader, 
              epochs, 
              model_loss_fn=None, 
              verbose=0, 
              SAVE_PATH=None,
              model_name=None,
              save_freq=5,
              device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    pred_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.003)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.2)

    metrics = {}    
    metrics['train'] = {}
    metrics['val'] = {}
    metrics['val']['best_acc'] = 0
    
    for epoch in (tqdm(range(int(epochs))) if verbose>0 else range(int(epochs))):
        # train for one epoch
        train_metrics = train_epoch(train_loader, 
                                       model,
                                       optimizer,
                                       pred_loss_fn, 
                                       model_loss_fn,
                                       device)        
        # evaluate on validation set
        val_metrics  = validate_epoch(val_loader, 
                                         model, 
                                         pred_loss_fn,
                                         model_loss_fn,
                                         device)
        
        for key, val in train_metrics.items():
            metrics['train'].setdefault(key, []).append(val)
        for key, val in val_metrics.items():
            metrics['val'].setdefault(key, []).append(val)
        scheduler.step()
        is_best = False
        if (val_metrics['acc'] > metrics['val']['best_acc']):
            metrics['val']['best_acc'] = max(val_metrics['acc'], metrics['val']['best_acc'])
            is_best = True
        if SAVE_PATH is not None:
            if (is_best or ((epoch % save_freq)==0)):
                save_checkpoint({
                    'model_name': model_name,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_acc': metrics['val']['best_acc'],
                    'metrics': metrics,
                }, is_best, model_name, SAVE_PATH, epoch)
        if verbose==2:
            print(f'Train acc: {train_metrics["acc"]}, Valid acc: {val_metrics["acc"]}')
    return metrics


def save_checkpoint(state, is_best, model_name, PATH, epoch=None):
    save_path = str(PATH)+'/'+str(model_name)+'_E_st'+str(epoch)+'_ckpnt.pth.tar'
    torch.save(state, save_path)
    if is_best:
        best_path = f'{PATH}/{model_name}_model_best.pth.tar'
        shutil.copyfile(save_path, best_path)
