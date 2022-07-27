import sys
import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
np.random.seed(1)
# import pandas as pd
import cv2
# from PIL import Image
# from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
# from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


from torch.utils.data import random_split, DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from dataloader import *

from setup import *

import warnings
warnings.filterwarnings('ignore')

############################
# Sensorium Stuff
from sensorium.utility import submission_lightning


# api_key = 'b508002bdc18b80b784941855ce5a0e722ef50d8'
# os.environ["WANDB_API_KEY"] = api_key
# wandb.init()

# Seeds
import random
random.seed(1)

import configparser

# Get the username of the person running the job, this is used for parsing the config
username = str(sys.argv[1])
print("Using username:", username, "to run.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    n_gpus = 1
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        
        print("GPUs detected: = {}".format( torch.cuda.device_count() ) )
        
        for i in range(n_gpus):
            print("_______")
            print( torch.cuda.get_device_name( i ) )
            print("_______")


    # pt_model = model_center_pt.SQUEEZENET_1_1()
    # pt_model = pt_model.load_from_checkpoint('/users/aarjun1/data/aarjun1/color_cnn/checkpoints/unet_vgg-epoch=56-val_loss=2.96.ckpt')

    # base = pt_model.slice

    config = configparser.ConfigParser()
    config.read("experiment_configuration.cfg")

    # Mode
    test_mode = config[username].getboolean('test_mode')
    val_mode = config[username].getboolean('val_mode')
    continue_training = config[username].getboolean('continue_training')
    
    # Training Method
    direct_training = config[username].getboolean('direct_training')
    fine_tuning = config[username].getboolean('fine_tuning')
    base_freeze = config[username].getboolean('base_freeze')
    pre_training = not(direct_training)

    # Hyper-Parameters
    prj_name = config[username]['prj_name']

    if direct_training:
        prj_name = prj_name + "_direct_training" #+ "_continued_continued" #_continued"
        n_neurons = 7776
        batch_size_per_gpu = 32
        # lr = 0.01

        if fine_tuning:
            n_neurons_list = [8372, 7344, 7334, 8107, 8098, 7776]
    else:
        prj_name = prj_name + "_pre_training" #+ "_continued"
        n_neurons_list = [8372, 7344, 7334, 8107, 8098, 7776]
        batch_size_per_gpu_train = 24
        batch_size_per_gpu_val = 128

    lr = config['DEFAULT'].getfloat('lr')

    weight_decay =  config[username].getfloat('weight_decay')
    hidden_size =  config[username].getint('hidden_size')
    timesteps = config[username].getint('timesteps')
    kernel_size = config[username].getint('kernel_size')
    pre_kernel_size = config[username].getint('pre_kernel_size')
    num_epochs = config[username].getint('num_epochs')

    VGG_bool = config[username].getboolean('VGG_bool')
    freeze_VGG = config[username].getboolean('freeze_VGG')
    HMAX_bool = config[username].getboolean('HMAX_bool')
    simple_ff_bool = config[username].getboolean('simple_ff_bool')
    sensorium_ff_bool = not(VGG_bool or HMAX_bool or simple_ff_bool)

    n_ori = config[username].getint("n_ori")
    n_scales = config[username].getint("n_scales")

    InT_bool = config[username].getboolean('InT_bool')
    batchnorm_bool = config[username].getboolean('batchnorm_bool')
    gaussian_bool = config[username].getboolean('gaussian_bool')

    visualize_bool = config[username].getboolean('visualize_bool')

    orthogonal_init = config[username].getboolean('orthogonal_init')
    exp_weight = config[username].getboolean('exp_weight')
    noneg_constraint = config[username].getboolean('noneg_constraint')
    clamp_weights = config[username].getboolean('clamp_weights')
    plot_weights = config[username].getboolean('plot_weights')

    corr_loss = config[username].getboolean('corr_loss')
    simple_to_complex = config[username].getboolean('simple_to_complex')
    simple_to_complex_gamma = config[username].getboolean('simple_to_complex_gamma')

    scale_image = config[username].getfloat('scale_image')

    # dataset_names = ['21067-10-18', '22846-10-16', '23343-5-17', '23656-14-22', '23964-4-22']

    # Calling the dataloader
    if direct_training:
        data = sensorium_loader_direct(batch_size_per_gpu, n_gpus)
    else:
        data = sensorium_loader_pretrain(batch_size_per_gpu_train, batch_size_per_gpu_val, n_gpus, scale_image)

    # Initializing the model
    if direct_training:
        if not fine_tuning:
            model = InT_Sensorium_Baseline_Direct(prj_name, lr, weight_decay, n_neurons, hidden_size, timesteps, kernel_size, \
                        pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                        exp_weight, noneg_constraint, visualize_bool, fine_tuning, \
                        dataloaders = data.dataloaders["train"], gaussian_bool = gaussian_bool, \
                        batch_size_per_gpu = batch_size_per_gpu, n_gpus = n_gpus, sensorium_ff_bool = sensorium_ff_bool)
        else:
            # print('HERREEE')
            base_model = InT_Sensorium_Baseline_Pretrain(prj_name, lr, weight_decay, n_neurons_list, hidden_size, timesteps, kernel_size, \
                        pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                        exp_weight, noneg_constraint, visualize_bool, gaussian_bool = gaussian_bool, \
                        sensorium_ff_bool = sensorium_ff_bool, clamp_weights = clamp_weights, plot_weights = plot_weights)
            model = InT_Sensorium_Baseline_Direct(prj_name, lr, weight_decay, n_neurons, hidden_size, timesteps, kernel_size, \
                        pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                        exp_weight, noneg_constraint, visualize_bool, fine_tuning, base_model, base_freeze, \
                        data.dataloaders["train"], gaussian_bool, batch_size_per_gpu, n_gpus, sensorium_ff_bool = sensorium_ff_bool)
    else:
        model = InT_Sensorium_Baseline_Pretrain(prj_name, lr, weight_decay, n_neurons_list, hidden_size, timesteps, kernel_size, \
                    pre_kernel_size, VGG_bool, freeze_VGG, InT_bool, batchnorm_bool, orthogonal_init, \
                    exp_weight, noneg_constraint, visualize_bool, data.dataloaders_train["train"], \
                    gaussian_bool, sensorium_ff_bool, clamp_weights, plot_weights, corr_loss, HMAX_bool, \
                    simple_to_complex, n_ori, n_scales, simple_ff_bool, simple_to_complex_gamma, scale_image)

    if test_mode or val_mode or continue_training:
        model = model.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/checkpoints/' + prj_name + '/sensorium-epoch=26-val_corr=0.3265847861766815-val_loss=13657369.0.ckpt')

        print('Loaded Checkpoint')

        ########################## While Testing ##########################
        ## Need to force change some variables after loading a checkpoint
        model.prj_name = prj_name + "_trained"

        model.lr = lr
        model.n_ori = n_ori
        model.n_scales = n_scales
        model.visualize_bool = visualize_bool
        model.fine_tuning = fine_tuning
        model.base_freeze = base_freeze
        model.HMAX_bool = HMAX_bool
        model.simple_to_complex = simple_to_complex
        model.simple_to_complex_gamma = simple_to_complex_gamma

        model.gaussian_bool = gaussian_bool
        model.sensorium_ff_bool = sensorium_ff_bool
        model.clamp_weights = clamp_weights
        model.plot_weights = plot_weights

        model.corr_loss = corr_loss

        if direct_training:
            model.dataloaders = data.dataloaders["train"]
        else:
            model.dataloaders = data.dataloaders_train["train"]
        model.gaussian_bool = gaussian_bool
    
        ###################################################################

    print(model)
    print('Number of Parameters', count_parameters(model))
    
    if continue_training:
        prj_name = prj_name + "_continued"


    # Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_corr",
                            dirpath="/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/"+username+"/checkpoints/" + prj_name,
                            filename="sensorium-{epoch}-{val_corr}-{val_loss}",
                            save_top_k=8,
                            mode="max",
                        )

    # wandb_logger = WandbLogger(project=prj_name)
    # # log gradients, parameter histogram and model topology
    # wandb_logger.watch(model, log="all")
    
    trainer = pl.Trainer(max_epochs = num_epochs, gpus=-1, accelerator = 'dp', callbacks = [checkpoint_callback]) #, gradient_clip_val= 0.5, \
                                                # gradient_clip_algorithm="value") #, logger = wandb_logger)
    # Train
    if not(test_mode or val_mode):
        trainer.fit(model, data)

        # Reading current config
        with open("experiment_configuration.cfg", "r") as input:
            with open("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/"+username+"/checkpoints/" + prj_name + "/experiment_configuration.cfg", "w") as output:
                for line in input:
                    output.write(line)
    # Val
    elif val_mode:
        trainer.validate(model, data) 
    # Test Sensorium
    else:

        data_key='26872-17-20'
        data.dataset_name = data_key

        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/submission_files", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)

        # generate the submission file
        submission_lightning.generate_submission_file(trained_model=model, 
                                    trainer_lightning = trainer,
                                    data_lightning = data,
                                    dataloaders=data.dataloaders if direct_training else data.dataloaders_val,
                                    data_key=data_key,
                                    path=job_dir + "/",
                                    prj_name = model.prj_name,
                                    device="cuda")

        # trainer.test(model, data)