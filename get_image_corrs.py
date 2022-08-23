import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataloader
from setup import *

filenames = ['/media/data_cifs/projects/prj_sensorium/arjun/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6']

dataset_fn = 'sensorium.datasets.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'batch_size': 16,
                 'scale':.25,
                 }

dataloaders = get_data(dataset_fn, dataset_config)

# Working with Sensorium data
dataset_name = '26872-17-20'

test_dataset = dataloaders['test'][dataset_name].dataset

#Setting Parameters
prj_name = "checkpoints_sensorium_SQZN11_5_InT_BN_track_all_sep_3t_5k_0003_GAP_6_datasets_gaussian_extra_activ_exc_scale_1"
test_mode = False
val_mode = True
continue_training = False

direct_training = False
fine_tuning = False
base_freeze = False

weight_decay = 1e-4
hidden_size = 128
timesteps = 3
kernel_size = 5
pre_kernel_size = 5
num_epochs = 1500

VGG_bool = True
freeze_VGG = False
HMAX_bool = False
simple_ff_bool = False
sensorium_ff_bool = False

n_ori = 12
n_scales = 9

InT_bool = True
batchnorm_bool = True
gaussian_bool = True
visualize_bool = False

orthogonal_init = True
exp_weight = False
noneg_constraint = False
clamp_weights = False
plot_weights = True

corr_loss = False
simple_to_complex = False
simple_to_complex_gamma = False

scale_image = 1

n_neurons = 7776
n_neurons_list = [8372, 7344, 7334, 8107, 8098]
batch_size_per_gpu_train = 32
batch_size_per_gpu_val = 128
lr = 0.003  

batch_size_per_gpu = 16
n_gpus = 1

data = sensorium_loader_direct(batch_size_per_gpu, n_gpus)

model = InT_Sensorium_Baseline_Pretrain(prj_name, lr, weight_decay, n_neurons_list, hidden_size, timesteps, kernel_size, \
                    pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                    exp_weight, noneg_constraint, visualize_bool, test_dataset, \
                    gaussian_bool, sensorium_ff_bool, clamp_weights, plot_weights, corr_loss, HMAX_bool, \
                    simple_to_complex, n_ori, n_scales, simple_ff_bool, simple_to_complex_gamma, scale_image)

model = model.load_from_checkpoint('/media/data_cifs/projects/prj_sensorium/old_modified/checkpoints/' + prj_name + '/sensorium-epoch=58-val_corr=0.30859655141830444-val_loss=13883785.0.ckpt')