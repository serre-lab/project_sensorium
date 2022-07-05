import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn import init
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
np.random.seed(1)
# import pandas as pd
import cv2
import _pickle as pickle
import math
# from PIL import Image
# from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torch.autograd import Variable

import torchvision.models as models

from collections import OrderedDict
import skimage.color as sic

from sklearn.decomposition import PCA

import pytorch_lightning as pl
# from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb

import scipy as sp
import h5py
from scipy.ndimage.filters import correlate

import argparse
import os
import random
import shutil
import time
import warnings

# Seeds

import random
random.seed(1)

from recurrent_circuits import FFhGRU
from sensorium.utility import plotting

from sensorium.models.utility import prepare_grid
from sensorium.models.readouts import MultipleFullGaussian2d

from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from neuralpredictors.utils import get_module_output

from dataloader import *

####################################################

from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def corr(
    y1: ArrayLike, y2: ArrayLike, axis: Union[None, int, Tuple[int]] = -1, eps: int = 1e-8, **kwargs
) -> np.ndarray:
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).
    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2
    Returns: correlation array
    """

    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)

####################################################

class InT_Sensorium_Baseline_Direct(pl.LightningModule):
    def __init__(self, prj_name, lr, weight_decay, n_neurons, hidden_size, timesteps, kernel_size, \
                 pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                 exp_weight, noneg_constraint, visualize_bool = False, fine_tuning = False, \
                 base_model = None, base_freeze = False, dataloaders = None, gaussian_bool = False, \
                 batch_size_per_gpu = None, n_gpus = None):
        super().__init__()
        
        self.parameter_dict = {'prj_name':prj_name, 'lr':lr, 'weight_decay':weight_decay, 'n_neurons':n_neurons, \
                               'hidden_size':hidden_size, 'timesteps':timesteps, 'kernel_size':kernel_size, 'pre_kernel_size':pre_kernel_size, \
                               'VGG_bool':VGG_bool, 'freeze_VGG':freeze_VGG, 'InT_bool':InT_bool, 'batchnorm_bool':batchnorm_bool, \
                               'orthogonal_init':orthogonal_init, 'exp_weight':exp_weight, 'noneg_constraint':noneg_constraint, \
                               'visualize_bool':visualize_bool, 'fine_tuning':fine_tuning, 'base_freeze':base_freeze, \
                               'gaussian_bool':gaussian_bool, 'batch_size_per_gpu':batch_size_per_gpu, 'n_gpus':n_gpus}

        print('self.parameter_dict : ',self.parameter_dict)

        self.prj_name = prj_name 

        self.batch_size_per_gpu = batch_size_per_gpu
        self.n_gpus = n_gpus

        self.dataloaders = sensorium_loader_direct(self.batch_size_per_gpu, self.n_gpus) #dataloaders
        self.dataloaders = self.dataloaders.dataloaders["train"]

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_neurons = n_neurons
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.pre_kernel_size = pre_kernel_size
        self.VGG_bool = VGG_bool
        self.freeze_VGG = freeze_VGG
        self.InT_bool = InT_bool
        self.batchnorm_bool = batchnorm_bool
        self.orthogonal_init = orthogonal_init
        self.exp_weight = exp_weight
        self.noneg_constraint = noneg_constraint
        self.visualize_bool = visualize_bool
        self.fine_tuning = fine_tuning
        self.base_freeze = base_freeze
        self.gaussian_bool = gaussian_bool

        if self.batchnorm_bool:
            # self.nl = F.softplus
            self.nl = F.elu
        else:
            # self.nl = F.sigmoid
            self.nl = F.elu

        ######################### While Fine_Tuning #######################
        # print('BASEEE : ', base_model)
        
        ###################################################################

        # if not self.fine_tuning:
        if False:
            self.recurrent_circuit = FFhGRU(self.hidden_size, timesteps=self.timesteps, \
                                kernel_size=self.kernel_size, nl=self.nl, input_size=3, \
                                output_size=3, l1=0., pre_kernel_size=self.pre_kernel_size, \
                                VGG_bool = self.VGG_bool, InT_bool = self.InT_bool, batchnorm_bool = self.batchnorm_bool, 
                                noneg_constraint = self.noneg_constraint, exp_weight = self.exp_weight, \
                                orthogonal_init = self.orthogonal_init, freeze_VGG = self.freeze_VGG)
        else:
            base_model = base_model.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/checkpoints/checkpoints_sensorium_VGG_pre_InT_BN_track_7t_9k_0003_multiple_gaussian_readout_pre_training/sensorium-epoch=28-val_loss=12217932.0.ckpt').recurrent_circuit
            
            ##################### Freezing Weights ############################
            if self.base_freeze:
                base_model.eval()
                for p_i, param in enumerate(base_model.parameters()):
                    param.requires_grad = False

            self.recurrent_circuit = base_model


        if not self.gaussian_bool:    

             self.regression = nn.Linear(self.hidden_size, self.n_neurons)
            # self.regression = nn.Linear(self.hidden_size*32*18, self.n_neurons)
        
        else:
            #######################################
            print('Going to Gaussian Readout')
            # Gaussian Readout
            self.data_key = '26872-17-20'
            init_mu_range = 0.3
            readout_bias = True
            init_sigma = 0.1
            gamma_readout = 0.0076
            gauss_type = 'full'
            grid_mean_predictor = {'type': 'cortex',
                                    'input_dimensions': 2,
                                    'hidden_layers': 1,
                                    'hidden_features': 30,
                                    'final_tanh': True}

            print('self.dataloaders : ',self.dataloaders)

            # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
            batch = next(iter(list(self.dataloaders.values())[0]))
            in_name, out_name = (
                list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
            )

            print('in_name : ',in_name)
            print('out_name : ',out_name)


            session_shape_dict = get_dims_for_loader_dict(self.dataloaders)
            print('session_shape_dict : ',session_shape_dict)
            for k, v in session_shape_dict.items():
                session_shape_dict[k]['images'] = torch.Size([v['images'][0], 3, v['images'][2], v['images'][3]])
            print('session_shape_dict stacked : ',session_shape_dict)
            n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
            input_channels = [v[in_name][1] for v in session_shape_dict.values()]
            in_shapes_dict = {k: get_module_output(self.recurrent_circuit, v[in_name])[1:] for k, v in session_shape_dict.items()}

            grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(grid_mean_predictor, self.dataloaders)

            print('in_shapes_dict : ',in_shapes_dict)
            print('n_neurons_dict : ',n_neurons_dict)
            print('input_channels : ',input_channels)
            print('source_grids : ',source_grids[self.data_key].shape)

            gaussian_readout = MultipleFullGaussian2d(
            in_shape_dict=in_shapes_dict,
            loader=self.dataloaders,
            n_neurons_dict=n_neurons_dict,
            init_mu_range=init_mu_range,
            bias=readout_bias,
            init_sigma=init_sigma,
            gamma_readout=gamma_readout,
            gauss_type=gauss_type,
            grid_mean_predictor=grid_mean_predictor,
            grid_mean_predictor_type=grid_mean_predictor_type,
            source_grids=source_grids,)

            print('self.gaussian_readout : ', gaussian_readout)

            self.gaussian_readout = gaussian_readout[self.data_key]



        self.dropout = nn.Dropout(p=0.4)
        
        # Val Loss
        self.val_losses = []
        self.min_loss = 1000
        self.correlations_list = []

        # Testing Sensorium
        self.test_neural_responses = []

        # log hyperparameters
        self.save_hyperparameters()


    def forward(self, x):

        recurrent_out = self.recurrent_circuit(x) # , time_steps_exc, time_steps_inh, xbn, weights_to_check

        if self.batchnorm_bool:
            recurrent_out = self.dropout(recurrent_out)

        if not self.gaussian_bool:
            recurrent_out = F.avg_pool2d(recurrent_out, (recurrent_out.shape[-2], recurrent_out.shape[-1]), 1)
            recurrent_out = recurrent_out.squeeze()

            # # recurrent_out = F.max_pool2d(recurrent_out, (2,2), 2)
            # # recurrent_out = recurrent_out.view(recurrent_out.shape[0], -1)

            reg_out = self.regression((recurrent_out))

        else:
            reg_out  = self.gaussian_readout(recurrent_out)

        # reg_out = F.relu(reg_out)
        reg_out = F.elu(reg_out) + 1

        return reg_out
    
    #pytorch lighning functions
    def configure_optimizers(self):
        print('lrrr  : ',self.lr)
        optimiser = torch.optim.Adam(self.parameters(), self.lr) #, weight_decay = self.weight_decay)
        return optimiser

    # # learning rate warm-up
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # update params
    #     optimizer.step(closure=optimizer_closure)

    #     # Manual Learning Rate Schedule // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


    def poison_loss(self, output, target, train, avg = False, eps=1e-12):
        
        # Akash: Changing numpy to torch for cuda tensor computations
        poisson_loss = output - target * torch.log(output + eps)

        if train:
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            return poisson_loss
        else:
            # Akash: Because we are not reshaping, axis changes
            # poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            poisson_loss = torch.mean(poisson_loss, axis = -1) if avg else torch.sum(poisson_loss, axis = -1)
            return poisson_loss

    def smooth_l1_loss(self, output, target, train, avg = True, eps=1e-12):
        
        # Akash: Changing numpy to torch for cuda tensor computations
        poisson_loss = F.smooth_l1_loss(output, target, reduce = False)

        if train:
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            return poisson_loss
        else:
            # Akash: Because we are not reshaping, axis changes
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            # poisson_loss = torch.mean(poisson_loss, axis = -1) if avg else torch.sum(poisson_loss, axis = -1)
            return poisson_loss


    def training_step(self, batch, batch_idx):
        
        images, target_neural_resp = batch.images, batch.responses

        h, w = images.shape[-2], images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 1, h, w)
            # For getting 3 channels such that we can use with pretrained feefdorward drives
            images = torch.stack([images, images, images], dim = 1)
            # Akash: No need to reshape here
            # target_neural_resp = target_neural_resp.reshape(-1)

            # Akash added this to align dimensionalities
            images = images.squeeze()

        ########################
        pred_neural_resp = self(images)

        ########################
        loss = self.poison_loss(pred_neural_resp, target_neural_resp, train = True)

        ########################
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 

    # def training_step_end(self, out_tar):

    #     loss = criterion(out_tar['output'], out_tar['target'])

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     self.log('train_loss', loss,on_step=False, on_epoch=True,prog_bar=True)
        
    #     return loss


    def validation_step(self, batch, batch_idx):
        
        images, target_neural_resp = batch.images, batch.responses

        # print('target_neural_resp : ',torch.max(target_neural_resp), ' :: ',torch.min(target_neural_resp), ' :: ',torch.mean(target_neural_resp))

        h, w = images.shape[-2], images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 1, h, w)
            # For getting 3 channels such that we can use with pretrained feefdorward drives
            images = torch.stack([images, images, images], dim = 1)
            # Akash: No need to reshape here
            # target_neural_resp = target_neural_resp.reshape(-1)

            # Akash: added this to align dimensionalities
            images = images.squeeze()
        ########################
        pred_neural_resp = self(images)

        ########################
        if self.visualize_bool:
            plotting.visualize_neural_plots(pred_neural_resp.clone(), target_neural_resp.clone(), self.prj_name)
            self.visualize_bool = False

        ########################
        if batch_idx == 0:
            print('########')
            print('pred std: ',torch.std(pred_neural_resp, dim = 0), ' ::: mean : ',torch.mean(pred_neural_resp), ' ::: min',torch.min(pred_neural_resp), ' ::: max',torch.max(pred_neural_resp))
            print('target std: ',torch.std(target_neural_resp, dim = 0), ' ::: mean : ',torch.mean(target_neural_resp), ' ::: min',torch.min(target_neural_resp), ' ::: max',torch.max(target_neural_resp))
            print('########')
        # pred_neural_resp = (pred_neural_resp) / (pred_neural_resp.std(dim=0, keepdim=True) + 1e-12)
        # target_neural_resp = (target_neural_resp) / (target_neural_resp.std(dim=0, keepdim=True) + 1e-12)
        correlations = corr(target_neural_resp.cpu().numpy(), pred_neural_resp.cpu().numpy(), axis=0)

        if np.any(np.isnan(correlations)):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations).mean() * 100
                )
            )
        correlations[np.isnan(correlations)] = 0

        self.correlations_list += [correlations]

        ########################
        val_loss = self.poison_loss(pred_neural_resp, target_neural_resp, train = True)

        ########################
        time.sleep(1)
        # Akash: Wrapping this in a list wrapper
        val_loss_list = [val_loss.cpu().tolist()]
        # val_loss_list = val_loss.cpu().tolist()
        self.val_losses += val_loss_list

        ###########################
        # val_loss = torch.mean(val_loss)

        return val_loss

    def validation_epoch_end(self,losses):

        self.correlations_list = [np.mean(c_list) for c_list in self.correlations_list]
        # print('self.correlations_list : ',self.correlations_list)
        print('self.correlations_list correlation : ',np.mean(self.correlations_list))

        #################################

        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.sum(losses)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        #################################
        result_summary = OrderedDict()
        result_summary["error" + "_mean"] = losses
        print(result_summary)

        ##################################
        self.log('val_loss', losses, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_corr', np.mean(self.correlations_list), on_step=False, on_epoch=True, prog_bar=True)

        # Reset Parameters
        self.val_losses = []
        self.correlations_list = []

    def test_step(self, batch, batch_idx):
        
        images, target_neural_resp = batch.images, batch.responses

        h, w = images.shape[-2], images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 1, h, w)
            # For getting 3 channels such that we can use with pretrained feefdorward drives
            images = torch.stack([images, images, images], dim = 1)
            # Akash: No need to reshape here
            # target_neural_resp = target_neural_resp.reshape(-1)

            # Akash: added this to align dimensionalities
            images = images.squeeze()
        ########################
        pred_neural_resp = self(images)

        ###########################
        # self.test_neural_responses = torch.cat((self.test_neural_responses, pred_neural_resp.cpu()), dim = 0)
        self.test_neural_responses.append(pred_neural_resp.cpu())

        ########################
        if self.visualize_bool:
            visualize_neural_plots(pred_neural_resp.clone(), target_neural_resp.clone(), self.prj_name)
            self.visualize_bool = False

        ########################
        val_loss = self.poison_loss(pred_neural_resp, target_neural_resp, train = True)

        ########################
        time.sleep(1)
        # Akash: Wrapping this in a list wrapper
        val_loss_list = [val_loss.cpu().tolist()]
        # val_loss_list = val_loss.cpu().tolist()
        self.val_losses += val_loss_list

        ###########################
        # val_loss = torch.mean(val_loss)

        return val_loss

    def test_epoch_end(self,losses):

        ###########################
        # Save pred_neural_resp
        self.test_neural_responses = torch.cat(self.test_neural_responses, dim = 0)
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/test_neural_responses", self.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "neural_responses.npy")
        np.save(file_name, self.test_neural_responses.numpy())

        print('self.test_neural_responses :',self.test_neural_responses.shape)

        #################################
        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.sum(losses)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        #################################
        result_summary = OrderedDict()
        result_summary["error" + "_mean"] = losses
        print(result_summary)

        ##################################
        self.log('test_loss', losses, on_step=False, on_epoch=True, prog_bar=True)

        # Reset Parameters
        self.val_losses = []
        self.test_neural_responses = []

    # def test_step(self, batch, batch_idx):



class InT_Sensorium_Baseline_Pretrain(pl.LightningModule):
    def __init__(self, prj_name, lr, weight_decay, n_neurons_list, hidden_size, timesteps, kernel_size, \
                 pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                 exp_weight, noneg_constraint, visualize_bool = False, dataloaders = None, gaussian_bool = False):
        super().__init__()
        
        self.parameter_dict = {'prj_name':prj_name, 'lr':lr, 'weight_decay':weight_decay, 'n_neurons':n_neurons_list, \
                               'hidden_size':hidden_size, 'timesteps':timesteps, 'kernel_size':kernel_size, 'pre_kernel_size':pre_kernel_size, \
                               'VGG_bool':VGG_bool, 'freeze_VGG':freeze_VGG, 'InT_bool':InT_bool, 'batchnorm_bool':batchnorm_bool, \
                               'orthogonal_init':orthogonal_init, 'exp_weight':exp_weight, 'noneg_constraint':noneg_constraint, \
                               'visualize_bool':visualize_bool, 'gaussian_bool':gaussian_bool}

        print('self.parameter_dict : ',self.parameter_dict)

        self.prj_name = prj_name 

        self.batch_size_per_gpu_train = 32
        self.batch_size_per_gpu_val = 128
        self.n_gpus = 8

        self.dataloaders = sensorium_loader_pretrain(self.batch_size_per_gpu_train, self.batch_size_per_gpu_val, self.n_gpus) #dataloaders
        self.dataloaders = self.dataloaders.dataloaders_train["train"]

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_neurons_list = n_neurons_list
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.pre_kernel_size = pre_kernel_size
        self.VGG_bool = VGG_bool
        self.freeze_VGG = freeze_VGG
        self.InT_bool = InT_bool
        self.batchnorm_bool = batchnorm_bool
        self.orthogonal_init = orthogonal_init
        self.exp_weight = exp_weight
        self.noneg_constraint = noneg_constraint
        self.visualize_bool = visualize_bool
        self.gaussian_bool = gaussian_bool

        if self.batchnorm_bool:
            # self.nl = F.softplus
            self.nl = F.elu
        else:
            # self.nl = F.sigmoid
            self.nl = F.elu

        ########################## While Testing ##########################
        
        ###################################################################

        print('Going to recurrent_circuit')

        self.recurrent_circuit = FFhGRU(self.hidden_size, timesteps=self.timesteps, \
                            kernel_size=self.kernel_size, nl=self.nl, input_size=3, \
                            output_size=3, l1=0., pre_kernel_size=self.pre_kernel_size, \
                            VGG_bool = self.VGG_bool, InT_bool = self.InT_bool, batchnorm_bool = self.batchnorm_bool, 
                            noneg_constraint = self.noneg_constraint, exp_weight = self.exp_weight, \
                            orthogonal_init = self.orthogonal_init, freeze_VGG = self.freeze_VGG)

            
        if not self.gaussian_bool:    
            regression_list = []

            for n_i, n_neurons in enumerate(self.n_neurons_list):
                regression_list.append(nn.Linear(self.hidden_size, n_neurons))

            self.regression_list = nn.ModuleList(regression_list)
            
            # self.regression = nn.Linear(self.hidden_size, self.n_neurons)
            # self.regression = nn.Linear(self.hidden_size*32*18, self.n_neurons)
        
        else:
            #######################################
            print('Going to Gaussian Readout')
            # Gaussian Readout
            init_mu_range = 0.3
            readout_bias = True
            init_sigma = 0.1
            gamma_readout = 0.0076
            gauss_type = 'full'
            grid_mean_predictor = {'type': 'cortex',
                                    'input_dimensions': 2,
                                    'hidden_layers': 1,
                                    'hidden_features': 30,
                                    'final_tanh': True}

            print('self.dataloaders : ',self.dataloaders)

            # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
            batch = next(iter(list(self.dataloaders.values())[0]))
            in_name, out_name = (
                list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
            )

            print('in_name : ',in_name)
            print('out_name : ',out_name)


            session_shape_dict = get_dims_for_loader_dict(self.dataloaders)
            print('session_shape_dict : ',session_shape_dict)
            for k, v in session_shape_dict.items():
                session_shape_dict[k]['images'] = torch.Size([v['images'][0], 3, v['images'][2], v['images'][3]])
            print('session_shape_dict stacked : ',session_shape_dict)
            n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
            input_channels = [v[in_name][1] for v in session_shape_dict.values()]
            in_shapes_dict = {k: get_module_output(self.recurrent_circuit, v[in_name])[1:] for k, v in session_shape_dict.items()}

            grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(grid_mean_predictor, self.dataloaders)

            print('in_shapes_dict : ',in_shapes_dict)
            print('n_neurons_dict : ',n_neurons_dict)
            print('input_channels : ',input_channels)
            print('source_grids : ',source_grids['21067-10-18'].shape)

            self.gaussian_readout = MultipleFullGaussian2d(
            in_shape_dict=in_shapes_dict,
            loader=self.dataloaders,
            n_neurons_dict=n_neurons_dict,
            init_mu_range=init_mu_range,
            bias=readout_bias,
            init_sigma=init_sigma,
            gamma_readout=gamma_readout,
            gauss_type=gauss_type,
            grid_mean_predictor=grid_mean_predictor,
            grid_mean_predictor_type=grid_mean_predictor_type,
            source_grids=source_grids,)

            # print('self.gaussian_readout : ',self.gaussian_readout['21067-10-18'])

            self.data_idx_to_key = {0: '21067-10-18', 1: '22846-10-16', 2: '23343-5-17', 3: '23656-14-22', 4: '23964-4-22'}

        # print(10/0)

        self.dropout = nn.Dropout(p=0.4)


        # Val Loss
        self.val_losses = []
        self.min_loss = 1000
        self.correlations_list = []

        # Testing Sensorium
        self.test_neural_responses = torch.empty(0)

        # log hyperparameters
        self.save_hyperparameters()


    def force_cudnn_initialization(self):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


    def forward(self, x, regressor_i):

        
        # # if self.conv_init:
        # #     self.force_cudnn_initialization()
        # #     self.conv_init = False
        
        recurrent_out = self.recurrent_circuit(x) # , time_steps_exc, time_steps_inh, xbn, weights_to_check

        recurrent_out = self.dropout(recurrent_out)

        if not self.gaussian_bool:
            recurrent_out = F.avg_pool2d(recurrent_out, (recurrent_out.shape[-2], recurrent_out.shape[-1]), 1)
            recurrent_out = recurrent_out.squeeze()

            # # recurrent_out = F.max_pool2d(recurrent_out, (2,2), 2)
            # # recurrent_out = recurrent_out.view(recurrent_out.shape[0], -1)

            reg_out = self.regression_list[regressor_i]((recurrent_out))

        else:
            reg_out  = self.gaussian_readout[self.data_idx_to_key[regressor_i]]((recurrent_out))

        # reg_out = F.relu(reg_out)
        reg_out = F.elu(reg_out) + 1

        return reg_out
    
    #pytorch lighning functions
    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), self.lr) #, weight_decay = self.weight_decay)
        return optimiser

    # # learning rate warm-up
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # update params
    #     optimizer.step(closure=optimizer_closure)

    #     # Manual Learning Rate Schedule // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


    def poison_loss(self, output, target, train, avg = False, eps=1e-12):

        # output = (output) / (output.std(dim=0, keepdim=True) + eps)
        
        # Akash: Changing numpy to torch for cuda tensor computations
        poisson_loss = output - target * torch.log(output + eps)

        if train:
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            return poisson_loss
        else:
            # Akash: Because we are not reshaping, axis changes
            # poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            poisson_loss = torch.mean(poisson_loss, axis = -1) if avg else torch.sum(poisson_loss, axis = -1)
            return poisson_loss

    def smooth_l1_loss(self, output, target, train, avg = True, eps=1e-12):
        
        # Akash: Changing numpy to torch for cuda tensor computations
        poisson_loss = F.smooth_l1_loss(output, target, reduce = False)

        if train:
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            return poisson_loss
        else:
            # Akash: Because we are not reshaping, axis changes
            poisson_loss = torch.mean(poisson_loss) if avg else torch.sum(poisson_loss)
            # poisson_loss = torch.mean(poisson_loss, axis = -1) if avg else torch.sum(poisson_loss, axis = -1)
            return poisson_loss


    def training_step(self, batch, batch_idx):

        images, target_neural_resp = [], []
        for i in range(len(batch)):
            images.append(batch[i].images)
            target_neural_resp.append(batch[i].responses)

        for i in range(len(images)):
            h, w = images[i].shape[-2], images[i].shape[-1]
            if len(images[i].shape) == 4:
                images[i] = images[i].reshape(-1, 1, h, w)
                # For getting 3 channels such that we can use with pretrained feefdorward drives
                images[i] = torch.cat([images[i], images[i], images[i]], dim = 1)
                # Akash: No need to reshape here
                # target_neural_resp = target_neural_resp.reshape(-1)

                # Akash added this to align dimensionalities
                # if images[i].shape[0] != 1:
                #     images[i] = images[i].squeeze()

        ########################
        pred_neural_resp = []
        for i in range(len(images)):
            pred_neural_resp.append(self(images[i], i))

        ########################
        losses = []
        for i in range(len(pred_neural_resp)):
            ########################
            # pred_neural_resp[i] = (pred_neural_resp[i]) / (pred_neural_resp[i].std(dim=0, keepdim=True) + 1e-12)
            # target_neural_resp[i] = (target_neural_resp[i]) / (target_neural_resp[i].std(dim=0, keepdim=True) + 1e-12)

            loss_i = self.poison_loss(pred_neural_resp[i], target_neural_resp[i], train = True)
            losses.append(loss_i)

        loss = torch.stack(losses, dim = 0)
        loss = torch.sum(loss)

        ########################
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 


    # def training_step_end(self, out_tar):

    #     loss = criterion(out_tar['output'], out_tar['target'])

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     self.log('train_loss', loss,on_step=False, on_epoch=True,prog_bar=True)
        
    #     return loss


    def validation_step(self, batch, batch_idx, dataset_idx):
        
        images, target_neural_resp = batch.images, batch.responses

        # print('images : ', images.shape)
        # print('dataset_idx : ', dataset_idx)
        # print('target_neural_resp std: ',torch.std(target_neural_resp, dim = 0), ' :: ',torch.min(target_neural_resp), ' :: ',torch.mean(target_neural_resp))

        h, w = images.shape[-2], images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 1, h, w)
            # For getting 3 channels such that we can use with pretrained feefdorward drives
            images = torch.cat([images, images, images], dim = 1)
            # Akash: No need to reshape here
            # target_neural_resp = target_neural_resp.reshape(-1)

            # Akash: added this to align dimensionalities
            # if images.shape[0] != 1:
            #         images = images.squeeze()
        ########################
        pred_neural_resp = self(images, dataset_idx)

        ########################
        if self.visualize_bool:
            plotting.visualize_neural_plots(pred_neural_resp.clone(), target_neural_resp.clone(), self.prj_name)
            self.visualize_bool = False

        ########################
        if batch_idx == 0:
            print('########')
            print('pred std: ',torch.std(pred_neural_resp, dim = 0), ' ::: mean : ',torch.mean(pred_neural_resp), ' ::: min',torch.min(pred_neural_resp), ' ::: max',torch.max(pred_neural_resp))
            print('target std: ',torch.std(target_neural_resp, dim = 0), ' ::: mean : ',torch.mean(target_neural_resp), ' ::: min',torch.min(target_neural_resp), ' ::: max',torch.max(target_neural_resp))
            print('########')
        # pred_neural_resp = (pred_neural_resp) / (pred_neural_resp.std(dim=0, keepdim=True) + 1e-12)
        # target_neural_resp = (target_neural_resp) / (target_neural_resp.std(dim=0, keepdim=True) + 1e-12)
        correlations = corr(target_neural_resp.cpu().numpy(), pred_neural_resp.cpu().numpy(), axis=0)

        if np.any(np.isnan(correlations)):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations).mean() * 100
                )
            )
        correlations[np.isnan(correlations)] = 0

        self.correlations_list += [correlations]

        ########################
        val_loss = self.poison_loss(pred_neural_resp, target_neural_resp, train = True)

        ########################
        time.sleep(1)
        # Akash: Wrapping this in a list wrapper
        val_loss_list = [val_loss.cpu().tolist()]
        # val_loss_list = val_loss.cpu().tolist()
        self.val_losses += val_loss_list

        ###########################
        # val_loss = torch.sum(val_loss)

        return val_loss

    def validation_epoch_end(self,losses):

        self.correlations_list = [np.mean(c_list) for c_list in self.correlations_list]
        # print('self.correlations_list : ',self.correlations_list)
        print('self.correlations_list correlation : ',np.mean(self.correlations_list))

        #################################
        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.sum(losses)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        #################################
        result_summary = OrderedDict()
        result_summary["error" + "_mean"] = losses
        print(result_summary)

        ##################################
        self.log('val_loss', losses, on_step=False, on_epoch=True, prog_bar=True)

        # Reset Parameters
        self.val_losses = []
        self.correlations_list = []

    def test_step(self, batch, batch_idx):
        
        images, target_neural_resp = batch.images, batch.responses

        h, w = images.shape[-2], images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 1, h, w)
            # For getting 3 channels such that we can use with pretrained feefdorward drives
            images = torch.stack([images, images, images], dim = 1)
            # Akash: No need to reshape here
            # target_neural_resp = target_neural_resp.reshape(-1)

            # Akash: added this to align dimensionalities
            images = images.squeeze()
        ########################
        pred_neural_resp = self(images)

        ###########################
        self.test_neural_responses = torch.cat((self.test_neural_responses, pred_neural_resp.cpu()), dim = 0)
        print('pred_neural_resp : ',pred_neural_resp.shape)
        print('test_neural_responses step :',self.test_neural_responses.shape)

        ########################
        if self.visualize_bool:
            visualize_neural_plots(pred_neural_resp.clone(), target_neural_resp.clone(), self.prj_name)
            self.visualize_bool = False

        ########################
        val_loss = self.poison_loss(pred_neural_resp, target_neural_resp, train = False)

        ########################
        time.sleep(1)
        # Akash: Wrapping this in a list wrapper
        # val_loss_list = [val_loss.cpu().tolist()]
        val_loss_list = val_loss.cpu().tolist()
        self.val_losses += val_loss_list

        ###########################
        val_loss = torch.mean(val_loss)

        return val_loss

    def test_epoch_end(self,losses):

        ###########################
        # Save pred_neural_resp
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/test_neural_responses", self.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "neural_responses.npy")
        np.save(file_name, self.test_neural_responses.numpy())

        print('self.test_neural_responses :',self.test_neural_responses.shape)

        #################################
        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.mean(losses)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        #################################
        result_summary = OrderedDict()
        result_summary["error" + "_mean"] = losses
        print(result_summary)

        ##################################
        self.log('test_loss', losses, on_step=False, on_epoch=True, prog_bar=True)

        # Reset Parameters
        self.val_losses = []

    # def test_step(self, batch, batch_idx):

        

    

        

    