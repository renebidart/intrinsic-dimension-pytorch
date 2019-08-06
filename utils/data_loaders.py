import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms




def mnist_loaders(PATH, bs=256, size=28, download=True):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root=PATH, train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=4)

    valset = torchvision.datasets.MNIST(root=PATH, train=False,
                                           download=download, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False, num_workers=4)
    return train_loader, val_loader




def cifar_loaders(bs, PATH, size=32, download=True):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=PATH, train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=4)

    valset = torchvision.datasets.CIFAR10(root=PATH, train=False,
                                           download=download, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False, num_workers=4)
    return train_loader, val_loader



class SimpleCIFAR(Dataset):
    """CIFAR10 optionally with some labels removed and imgs resized"""

    def __init__(self, 
                 files_df, 
                 n_labels=None,
                 augment=None,
                 size=32,
                 return_loc=False,
                 base_path=None,
                 path_colname='path',
                 label_colname='label'):
        self.files = files_df
        self.size = size
        self.return_loc = return_loc
        self.base_path = base_path
        self.path_colname = path_colname
        self.label_colname = label_colname
        if augment is not None:
            self.transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.Resize(self.size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                       (0.2023, 0.1994, 0.2010))
                                                 ])
        else:
            self.transforms = transforms.Compose([transforms.Resize(self.size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                       (0.2023, 0.1994, 0.2010))
                                                  ])
        if isinstance(n_labels, int):
            max_label = np.sort(self.files[self.label_colname].unique())[int(n_labels)-1]
            self.files = self.files.loc[self.files[self.label_colname] <= int(max_label)]
            print(f'Creating dataloader with {n_labels} labels')
                
    def __getitem__(self, index):
        loc = self.files[self.path_colname].iloc[index]
        if self.base_path:
            loc = str(self.base_path) +'/'+ loc
            
        if self.label_colname:
            label = self.files[self.label_colname].iloc[index]
        else:
            label = 0
            
        img = Image.open(loc)
        img = self.transforms(img)
        
        if self.return_loc:
            return img, label, loc
        else:
            return img, label

    def __len__(self):
        return len(self.files)

        
def make_gen_SimpleCIFAR(files_dict_loc, 
                         batch_size=256, 
                         num_workers=4,
                         n_labels=None,
                         augment=None,
                         size=32,
                         return_loc=False,
                         base_path=None,
                         path_colname='path',
                         label_colname='label'):                         

    folders = ['train', 'val']
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)

    datasets = {x: SimpleCIFAR(files_dict[x],
                                n_labels=n_labels,
                                augment=augment,
                                size=size,
                                return_loc=return_loc,
                                base_path=None,
                                path_colname='path',
                                label_colname='label')
                for x in folders
               }
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                   for x in folders
                  }
    return dataloaders
