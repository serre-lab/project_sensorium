import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer

########################################################################


import os

basepath = "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/data/"

# as filenames, we'll select all 7 datasets
filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]

dataset_fn = 'sensorium.datasets.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'batch_size': 128,
                 'scale':.25,
                 }

dataloaders = get_data(dataset_fn, dataset_config)
print(dataloaders)

########################################################################


model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
model_config = {'pad_input': False,
  'layers': 4,
  'input_kern': 9,
  'gamma_input': 6.3831,
  'gamma_readout': 0.0076,
  'hidden_kern': 7,
  'hidden_channels': 64,
  'depth_separable': True,
  'grid_mean_predictor': {'type': 'cortex',
   'input_dimensions': 2,
   'hidden_layers': 1,
   'hidden_features': 30,
   'final_tanh': True},
  'init_sigma': 0.1,
  'init_mu_range': 0.3,
  'gauss_type': 'full',
  'shifter': False,
  'stack': -1,
}

model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=42,)

########################################################################

print('########################################################################')
print(model)
print('########################################################################')

########################################################################

trainer_fn = "sensorium.training.standard_trainer"

trainer_config = {'max_iter': 200,
                 'verbose': True,
                 'lr_decay_steps': 4,
                 'avg_loss': False,
                 'lr_init': 0.009,
                 'track_training':True
                 }

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)
print('Trainer is ready to go')

########################################################################

validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
torch.save(model.state_dict(), '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/checkpoints/generalization_model.pth')

########################################################################