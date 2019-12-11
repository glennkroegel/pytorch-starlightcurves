import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os
from collections import defaultdict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    y = y[labels]
    return y

def collate_ts(data, device=torch.device('cpu')):
        # batch = []
        # for i in data:
        #     batch.append(i[1])
        batch = torch.stack(data)
        # if len(batch.size()) < 3:
        #     batch = batch.unsqueeze(0)
        bs = batch.size(0)
        sl = batch.size(2)
        ts = batch[:, 0]
        ys = batch[:, 1].unsqueeze(-1)
        mask_train = (ys != 0).float()
        y_train = ys * mask_train
        y_pred = ys
        batch_dict = {'observed_data': y_train.to(device), 
                      'observed_tp': ts[0].view(-1).to(device), 
                      'data_to_predict': y_pred.to(device), 
                      'tp_to_predict': ts[0].view(-1).to(device), 
                      'observed_mask': mask_train.to(device), 
                      'mask_predicted_data': mask_train.to(device), 
                      'labels': None, 
                      'mode': 'interp', 
                      'labels': None}
        batch_dict = batchify(batch_dict)
        return batch_dict

def batchify(data_dict):
    # Make the union of all time points and perform normalization across the whole dataset
    batch_dict = defaultdict(None)

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict["observed_data"],(0,2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

    batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    non_missing_tp = torch.sum(data_dict["data_to_predict"],(0,2)) != 0.
    batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]
    batch_dict["labels"] = data_dict["labels"]
    return batch_dict

def increase_resolution(batch, T=1000):
    batch = torch.stack(batch)
    bs = batch.size(0)
    ts = batch[:, 0]
    ts = ts - ts[:, 0].view(bs, 1)
    ys = batch[:, 1]
    mask = ~torch.isnan(ys).unsqueeze(-1).float()
    ti = ts[:, 0]
    tf = ts[:, -1]
    dt = (tf - ti)/T
    dt = dt.view(bs, 1)
    ts_interp = F.pad(dt.unsqueeze(1), (0, T), mode='replicate').squeeze()
    ts_interp[:, 0] = ti
    ts_interp = torch.cumsum(ts_interp, dim=1)
    return ts, ys, mask, ts_interp

def accuracy_from_logits(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    # inp = input.view(n, -1)
    inp = input.argmax(dim=-1).view(-1)
    # targs = targs.argmax(dim=-1).view(-1)
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