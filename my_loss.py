import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
from math import exp
from torch.autograd import Variable


def PSNR(output_image, true_image, max_value=1.0):
    mse = F.mse_loss(output_image, true_image)
    if mse == 0:
        return float("inf")

    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr.item()


class mse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self,  output, target):
        return self.loss(output, target)
