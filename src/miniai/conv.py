# AUTOGENERATED FROM 06_convolutions.ipynb
import pickle, gzip, math, os, time, shutil, torch, matplotlib as mpl, numpy as np
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from torch import tensor
from torch import nn

from torch.utils.data import DataLoader, default_collate
from typing import Mapping

try:
    from .training import *
    from .datasets import *
except:
    from src.miniai.training import *
    from src.miniai.datasets import *

# Creating conv function to return a customized conv layer
def conv(ni, nf, ks=3, stride=2, act=True):
    res = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res

# check for device, and add a tensor object or dictionary of tensor objects to that device
def_device = "mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
# def_device = "cpu"

def to_device(x, device=def_device):
    # Map dictionary values to the device if this format is present
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    # else map the tensor object to device
    return type(x)(o.to(device) for o in x)

# collate to call on batch grab
def collate_device(b): return to_device(default_collate(b))

__all__ = ['conv', 'to_device', 'collate_device', 'def_device']