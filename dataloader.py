import nnfabrik
from nnfabrik.builder import get_data, get_model, get_trainer
# import sensorium.datasets

import pandas as pd
import seaborn as sns

from lipstick import GifMaker
from mpl_toolkits import mplot3d

import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torch
torch.manual_seed(1)
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import os
import cv2
import numpy as np
np.random.seed(1)
import random
import scipy as sp
import matplotlib.pyplot as plt
import skimage.color as sic
import pickle
from collections import OrderedDict

# Seeds

import random
random.seed(1)

'''
filenames = ['/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']

The datasets 26872-17-20 (Sensorium) 27204-5-13 (Sensorium+) are different from the 5 other full datasets.
'''


class sensorium_loader_direct(pl.LightningDataModule):
    def __init__(self, batch_size_per_gpu, n_gpus, dataset_name = None):
        super().__init__()
          
        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
        
        # Sensorium dataloader
        if dataset_name == None:
            self.filenames = ['/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']
            # self.filenames = ['/media/data_cifs/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']
        else:
            self.filenames = ['/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']
            
            # self.filenames = ['/media/data_cifs/projects/prj_sensorium/arjun/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            #                 '/media/data_cifs/projects/prj_sensorium/arjun/data/static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']

        self.dataset_fn = 'sensorium.datasets.static_loaders'
        # self.dataset_config = {'paths': self.filenames,
        #                 'normalize': True,
        #                 'include_behavior': True,
        #                 'include_eye_position': True,
        #                 'batch_size': self.batch_size_train,
        #                 'exclude': None,
        #                 'file_tree': True,
        #                 'scale': 1,
        #                 'add_behavior_as_channels': False,
        #                 }

        self.dataset_config = {'paths': self.filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'batch_size': self.batch_size,
                 'scale':.25,
                 }

        # Directory to load Data of some mouse
        if dataset_name == None:
            self.dataset_name = '26872-17-20'
        else:
            self.dataset_name = dataset_name

        # Getting the dataloader
        print('Getting the dataloader')
        self.dataloaders = get_data(self.dataset_fn, self.dataset_config)
        print('Got the dataloader')

        self.test_tier = None
  
    # def setup(self, stage=None):
  

    def train_dataloader(self):

        # Setting up num of workers
        self.dataloaders['train'][self.dataset_name].num_workers = 0
        # Dropping last
        # self.dataloaders['train'][self.dataset_name].multiprocessing_context='spawn'
        
        # Return train_dataloader
        return self.dataloaders['train'][self.dataset_name]
  
    def val_dataloader(self):

        # Setting up num of workers
        self.dataloaders['validation'][self.dataset_name].num_workers = 0
        # Dropping last
        # self.dataloaders['validation'][self.dataset_name].multiprocessing_context='spawn'

        outt = self.dataloaders['validation'][self.dataset_name]

        # Return val_dataloader
        return outt
  
    def test_dataloader(self):
        
        # Setting up num of workers
        self.dataloaders[self.test_tier][self.dataset_name].num_workers = 0
        # Dropping last
        # self.dataloaders['test'][self.dataset_name].multiprocessing_context='spawn'
        
        # Return test_dataloader
        return self.dataloaders[self.test_tier][self.dataset_name]


class sensorium_loader_pretrain(pl.LightningDataModule):
    def __init__(self, batch_size_per_gpu_train, batch_size_per_gpu_val, n_gpus, scale_images):
        super().__init__()
          
        # Batch Size
        self.batch_size_train = n_gpus*batch_size_per_gpu_train
        self.batch_size_val = n_gpus*batch_size_per_gpu_val
        self.scale_images = scale_images
        
        # Sensorium dataloader
        self.filenames = ['/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']

        # self.filenames = ['/media/data_cifs/projects/prj_sensorium/arjun/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        #     '/media/data_cifs/projects/prj_sensorium/arjun/data/static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        #     '/media/data_cifs/projects/prj_sensorium/arjun/data/static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
        #     '/media/data_cifs/projects/prj_sensorium/arjun/data/static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
        #     '/media/data_cifs/projects/prj_sensorium/arjun/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
        #     '/media/data_cifs/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']

        self.dataset_fn = 'sensorium.datasets.static_loaders'
        # self.dataset_config = {'paths': self.filenames,
        #                 'normalize': True,
        #                 'include_behavior': True,
        #                 'include_eye_position': True,
        #                 'batch_size': self.batch_size_train,
        #                 'exclude': None,
        #                 'file_tree': True,
        #                 'scale': 1,
        #                 'add_behavior_as_channels': False,
        #                 }

        self.dataset_config = {'paths': self.filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'batch_size': self.batch_size_train,
                 'scale': self.scale_images,
                 }
        

        # Directory to load Data of some mouse
        self.dataset_names = ['21067-10-18', '22846-10-16', '23343-5-17', '23656-14-22', '23964-4-22', '26872-17-20']

        # Getting the dataloader
        print('Getting the train dataloader')
        self.dataloaders_train = get_data(self.dataset_fn, self.dataset_config)
        print('Got the train dataloader')

        print('Getting the val dataloader')
        self.dataset_config['batch_size'] = self.batch_size_val
        self.dataloaders_val = get_data(self.dataset_fn, self.dataset_config)
        print('Got the train dataloader')

        self.test_tier = None
  
    # def setup(self, stage=None):
  

    def train_dataloader(self):

        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        dataloaders_list = []

        for d_i, dataset_name in enumerate(self.dataset_names):
            # Setting up num of workers
            self.dataloaders_train['train'][dataset_name].num_workers = 0
            # Dropping last
            # self.dataloaders['train'][self.dataset_name].multiprocessing_context='spawn'

            dataloaders_list.append(self.dataloaders_train['train'][dataset_name])
        
        # Return train_dataloader
        return dataloaders_list
  
    # @pl.data_loader
    def val_dataloader(self):

        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        dataloaders_list = []

        for d_i, dataset_name in enumerate(self.dataset_names):
            # Setting up num of workers
            self.dataloaders_val['validation'][dataset_name].num_workers = 0
            # Dropping last
            # self.dataloaders['train'][self.dataset_name].multiprocessing_context='spawn'

            dataloaders_list.append(self.dataloaders_val['validation'][dataset_name])
        
        # Return train_dataloader
        return dataloaders_list
  
    def test_dataloader(self):
        
        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        dataloaders_list = []

        for d_i, dataset_name in enumerate(self.dataset_names):
            # Setting up num of workers
            if d_i == 5:
                self.dataloaders_val[self.test_tier][dataset_name].num_workers = 0
            else:
                self.dataloaders_val['validation'][dataset_name].num_workers = 0
            # Dropping last
            # self.dataloaders['train'][self.dataset_name].multiprocessing_context='spawn'
            
            if d_i == 5:
                dataloaders_list.append(self.dataloaders_val[self.test_tier][dataset_name])
            else:
                dataloaders_list.append(self.dataloaders_val['validation'][dataset_name])
        
        # Return train_dataloader
        return dataloaders_list