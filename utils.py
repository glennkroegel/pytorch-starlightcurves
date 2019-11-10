import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy_from_logits(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    inp = input.argmax(dim=-1).view(-1)
    # targs = targs.view(n, -1)
    ix1 = (targs == 0).nonzero().squeeze()
    ix2 = (targs == 1).nonzero().squeeze()
    ix3 = (targs == 2).nonzero().squeeze()
    acc1 = (inp[ix1] == targs[ix1]).float().mean().item()
    acc2 = (inp[ix2] == targs[ix2]).float().mean().item()
    acc3 = (inp[ix3] == targs[ix3]).float().mean().item()
    acc = (inp==targs).float().mean()
    return acc, acc1, acc2, acc3

def accuracy(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    inp = input.argmax(dim=-1).view(-1)
    targs = targs.view(n,-1)
    acc = (inp==targs).float().mean()
    return acc

def bce_acc(input, targs):
    bs = targs.shape[0]
    inp = torch.round(input).view(bs, 1)
    targs = targs.view(bs, 1)
    ix0 = (targs == 0).nonzero().squeeze()
    ix1 = (targs == 1).nonzero().squeeze()
    acc0 = (inp[ix0] == targs[ix0]).float().mean().item()
    acc1 = (inp[ix1] == targs[ix1]).float().mean().item()
    acc = (inp==targs).float().mean().item()
    return acc, acc0, acc1