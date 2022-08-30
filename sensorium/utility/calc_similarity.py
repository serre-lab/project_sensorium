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

# Squared Euclideans is the Dissimlarity Measure
def calc_rdms_torch(measurements):

    sum_sq_measurements = torch.sum(measurements**2, dim=1, keepdim=True)
    # Doing it the way ||x2 - x1||^2 = ||x2||^2 + ||x1||^2 - 2 <x2, x1> ----> Same thing goes for the y coordinates
    rdm = sum_sq_measurements + sum_sq_measurements.t() - 2 * torch.matmul(measurements, measurements.t())

    return rdm

def upper_triangle(A: torch.Tensor, offset=1) -> torch.Tensor:
    """Get the upper-triangular elements of a square matrix.
    :param A: square matrix
    :param offset: number of diagonals to exclude, including the main diagonal. Deafult is 1.
    :return: a 1-dimensional torch Tensor containing upper-triangular values of A
    """

    # print('inside upt')

    if (A.ndim != 2) or (A.size()[0] != A.size()[1]):
        raise ValueError("A must be square")

    i, j = torch.triu_indices(*A.size(), offset=offset, device=A.device)

    # print('outside upt')

    return A[i, j]

def cov(m):
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.shape[-1] - 1)  # 1 / N
    m -= torch.mean(m, dim=(1, 2), keepdim=True)
    mt = torch.transpose(m, 1, 2)  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def compute_rank_correlation(x, y):
    if len(x.shape) == 1 and len(y.shape) == 1:
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
    x, y = rankmin(x), rankmin(y)
    return corrcoef(x, y)


def corrcoef(x, y):
    batch_size = x.shape[0]
    x = torch.stack((x, y), 1)
    # calculate covariance matrix of rows
    c = cov(x)
    # normalize covariance matrix
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.pow(d, 0.5)
    stddev = stddev.repeat(1, 2).view(batch_size, 2, 2)
    c = c.div(stddev)
    c = c.div(torch.transpose(stddev, 1, 2))
    return c[:, 1, 0]


# DREW's METHOD

from torch.nn import functional as F

def pdist_rdm(x):
    return F.pdist(x)

def pearson_dist(x, y, mode="correlation"):
    d = torch.cdist(x, y, mode)
    return d