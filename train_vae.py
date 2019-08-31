import os
import sys
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from models.vae import SparseVAE
from utils.training_func_vae import train_vae
from utils.data_loaders import make_generators_MNIST_CTRNFS

parser = argparse.ArgumentParser()
parser.add_argument('--SAVE-PATH', type=str,
                    default='./output/train')
parser.add_argument('--files-dict-loc', type=str,
                    default='/media/rene/data/MNIST/files_dict.pkl')

parser.add_argument('--bs', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=50, 
                    help='number of epochs to train (default: 50)')
parser.add_argument('--rot-aug', action='store_true', default=False,
                    help='use rotation augmentation during training')
parser.add_argument('--d', type=int, default=1024, 
                    help='dimension for intrinsic dimension')
parser.add_argument('--latent-size', type=int, default=8, 
                    help='latent dim of vae')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=101, metavar='S',
                    help='random seed (default: 101)')
args = parser.parse_args()

def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if args.cuda else "cpu")
    img_size=40

    if args.rot_aug:
        train_loader, val_loader = make_generators_MNIST_CTRNFS(files_dict_loc=args.files_dict_loc,
                                                                batch_size=args.bs,
                                                                return_size=img_size,
                                                                rotation_range=(0, 360))
        rot_name = 'rot'
    else:
        train_loader, val_loader = make_generators_MNIST_CTRNFS(files_dict_loc=args.files_dict_loc,
                                                                batch_size=args.bs,
                                                                return_size=img_size,
                                                                rotation_range=None)
        rot_name = 'no_rot'
        
    model_name = f'SparseVAE_E{args.epochs}_l{args.latent_size}_d{args.d}_{rot_name}'
    print(f"Training {model_name}")
    
    model = SparseVAE(d=args.d, 
                      latent_size=args.latent_size, 
                      img_size=img_size).to(DEVICE)
    metrics = train_vae(model,
                        train_loader, 
                        val_loader, 
                        epochs=args.epochs, 
                        verbose=1, 
                        device=DEVICE,
                        SAVE_PATH=args.SAVE_PATH,
                        model_name=model_name)
    
    print(f"{model_name} best loss: {metrics['val']['best_loss']}")
    
#     loc = Path(args.SAVE_PATH / f"SparseVAE_e_{args.epochs}_l{args.latent_size}_d{args.d}_{rot_name}_metrics.pkl")
#     pickle.dump(metrics, open(loc, "wb"))

if __name__ == '__main__':
    main(args)
