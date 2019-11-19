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

def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    y = y[labels]
    return y

def accuracy_from_logits(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    # inp = input.view(n, -1)
    inp = input.argmax(dim=-1).view(n, -1)
    targs = targs.argmax(dim=-1).view(n, -1)
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

def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result