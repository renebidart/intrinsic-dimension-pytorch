import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn.functional as F


def mnist_loaders(PATH, bs=256, size=28, download=True, aug=False):
    if aug: 
        transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    
    else:
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


class MNISTCustomTRNFS(Dataset):
    def __init__(self, files_df, base_path=None, 
                 return_size=None, rotation_range=None, normalize=True,
                 path_colname='path', label_colname='label', 
                 select_label=None, return_loc=False, return_rot=False):
        """ MNIST:  Rotations without squishing or cutting off imgs.  
        
        Do transform, adjustable padding to return_size, then normalize
        
        files_df: Pandas Dataframe containing the class and path of an image
        base_path: the path to append to the filename
        
        return_size: final img size
        rotation_range: (tuple) range of degrees to rotate. if both equal, a fixed rotation.
        normalize: 
        path_colname: Name of column containing locations or filenames
        label_colname: Name of column containing labels
        select_label: if you only want one label returned
        return_loc: return location as well as the image and class
        """
        
        self.files = files_df
        self.base_path = base_path
        self.path_colname = path_colname
        self.label_colname = label_colname
        self.return_loc = return_loc
        self.return_size = return_size
        self.rotation_range = rotation_range
        self.normalize = normalize
        self.return_rot = return_rot

        if isinstance(select_label, int):
            self.files = self.files.loc[self.files[self.label_colname] == int(select_label)]
            print(f'Creating dataloader with only label {select_label}')
            
    def pad_to_size(self, img, new_size):
        delta_width = new_size - img.size()[1]
        delta_height = new_size - img.size()[2]
        pad_width = delta_width //2
        pad_height = delta_height //2
        img = F.pad(img, (pad_height, pad_height, pad_width, pad_width), 'constant', 0)
        return img

    def __getitem__(self, index):
        if self.base_path:
            loc = str(self.base_path) +'/'+ self.files[self.path_colname].iloc[index]
        else:
            loc = self.files[self.path_colname].iloc[index]
        if self.label_colname:
            label = self.files[self.label_colname].iloc[index]
        else:
            label = 0
            
        img = Image.open(loc)
        if self.rotation_range:
            random_angle = random.randint(self.rotation_range[0], self.rotation_range[1])
            img = transforms.RandomRotation((random_angle, random_angle), expand=True)(img)
            
        img = transforms.ToTensor()(img)
        if self.return_size:
            img = self.pad_to_size(img, self.return_size)
        if self.normalize:
            img = transforms.Normalize((0.1307,), (0.3081,))(img)# this is wrong norm because imgs are padded
            
        if self.return_rot:
            return img, label, random_angle

        if self.return_loc:
            return img, label, loc
        else:
            return img, label

    def __len__(self):
        return len(self.files)
    


def make_generators_MNIST_CTRNFS(files_dict_loc, batch_size, num_workers=2, 
                                 return_size=40, rotation_range=None, normalize=True,
                                 path_colname='path', label_colname='class', label=None, 
                                 return_loc=False, return_rot=False):
    
    """Uses augmentation for both training and validation!"""
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)

    folders = ['train', 'val']
    datasets = {}
    dataloaders = {}
    datasets = {x: MNISTCustomTRNFS(files_dict[x], base_path=None, 
                                       return_size=return_size, rotation_range=rotation_range, normalize=normalize,
                                       path_colname=path_colname, label_colname=label_colname, 
                                       select_label=label, return_loc=return_loc, return_rot=return_rot)
                for x in folders
               }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                   for x in folders
                  }
    return dataloaders['train'], dataloaders['val']




















########### DELETE ###################

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
