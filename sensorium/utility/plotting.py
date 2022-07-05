import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import os,sys
import numpy as np
np.random.seed(1)
import scipy as sp
import skimage.color as sic
import seaborn as sns
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Seeds
import random
random.seed(1)

def plot_weight_histo(weights, name, prj_name):

    ##########################
    job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/visualize_neural_plots", prj_name)
    os.makedirs(job_dir, exist_ok=True)

    ##########################
    weights = np.sort(weights.numpy())

    ##########################
    plt.figure()
    # edges = [-1/i for i in range(2, 20)] + [0] + [1/i for i in range(2, 20)][::-1]
    plt.hist(weights, bins = 'auto')
    plt.xlabel("Weights")
    plt.ylabel("Counts")
    plt.show()
    plt.savefig(os.path.join(job_dir, name.replace('.','_')))
    plt.close()

    # print('Weight Plotted')

def visualize_neural_plots(preds, targets, prj_name):

    ##########################
    job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/visualize_neural_plots", prj_name)
    os.makedirs(job_dir, exist_ok=True)

    ##########################
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    ##########################
    count = 0
    for pred, target in zip(preds, targets):
        plt.figure()
        plt.plot(pred, label = 'Pred', alpha=0.6)
        plt.plot(target, label = 'Target', alpha=0.6)
        plt.legend()
        plt.xlabel("Response")
        plt.ylabel("Neurons")
        plt.show()
        plt.savefig(os.path.join(job_dir, str(count)))
        plt.pause(1)
        plt.close()

        count += 1

        if count > 10:
            break

    # print('Weight Plotted')