import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    y = y[labels]
    return y

def collate_ts(data, device=device):
    batch = torch.stack(data)
    bs = batch.size(0)
    sl = batch.size(2)
    ts = batch[:, 0]
    y_obs = batch[:, 1].unsqueeze(-1)
    y_truth = batch[:, 2].unsqueeze(-1)
    mask_train = (y_obs != 0).float()
    mask_truth = (y_truth != 0).float()
    y_obs = y_obs * mask_train
    batch_dict = {'observed_data': y_obs.to(device), 
                    'observed_tp': ts[0].view(-1).to(device), 
                    'data_to_predict': y_truth.to(device), 
                    'tp_to_predict': ts[0].view(-1).to(device), 
                    'observed_mask': mask_train.to(device), 
                    'mask_predicted_data': mask_truth.to(device), 
                    'labels': None, 
                    'mode': 'interp', 
                    'labels': None}
    # batch_dict = batchify(batch_dict)
    return batch_dict

def collate_extrap(data, device=device):
    batch = torch.stack(data)
    bs = batch.size(0)
    sl = batch.size(-1)
    t1 = batch[:, 0]
    t2 = batch[:, 1]
    y1 = batch[:, 2].unsqueeze(-1)
    y2 = batch[:, 3].unsqueeze(-1)
    mask1 = (y1 != 0).float()
    mask2 = (y2 != 0).float()
    batch_dict = {'observed_data': y1.to(device), 
                    'observed_tp': t1[0].view(-1).to(device), 
                    'data_to_predict': y2.to(device), 
                    'tp_to_predict': t2[0].view(-1).to(device), 
                    'observed_mask': mask1.to(device), 
                    'mask_predicted_data': mask2.to(device), 
                    'labels': None, 
                    'mode': 'extrap', 
                    'labels': None}
    # batch_dict = batchify(batch_dict)
    return batch_dict

def collate_interp_sparse(data, device=device):
    '''No subsampling of obs measurement'''
    if not isinstance(data, torch.Tensor):
        batch = torch.stack(data)
    else:
        batch = data
    bs = batch.size(0)
    sl = batch.size(2)
    ts = batch[:, 0]
    y = batch[:, 1].unsqueeze(-1)
    mask = (y != 0).float()
    batch_dict = {'observed_data': y.to(device), 
                    'observed_tp': ts[0].view(-1).to(device), 
                    'data_to_predict': y.to(device), 
                    'tp_to_predict': ts[0].view(-1).to(device), 
                    'observed_mask': mask.to(device), 
                    'mask_predicted_data': mask.to(device), 
                    'labels': None, 
                    'mode': 'interp', 
                    'labels': None}
    # batch_dict = batchify(batch_dict)
    return batch_dict

def collate_2d(data, device=device):
    '''No subsample, 2d interpolation.'''
    if not isinstance(data, torch.Tensor):
        batch = torch.stack(data)
    else:
        batch = data
    bs = batch.size(0)
    sl = batch.size(-1)
    ts = batch[:, 0][0]
    y = batch[:, 1:].permute(0, 2, 1)
    y[torch.isnan(y)] = 0.
    mask = (y != 0).float()
    batch_dict = {'observed_data': y.to(device), 
                    'observed_tp': ts.view(-1).to(device), 
                    'data_to_predict': y.to(device), 
                    'tp_to_predict': ts.view(-1).to(device), 
                    'observed_mask': mask.to(device), 
                    'mask_predicted_data': mask.to(device), 
                    'labels': None, 
                    'mode': 'interp', 
                    'labels': None}
    # batch_dict = batchify(batch_dict)
    return batch_dict

def batchify(data_dict):
    # Make the union of all time points and perform normalization across the whole dataset
    batch_dict = {k:None for k in data_dict.keys()}

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

def process_gaia_csv(infile, min_count=15):
    df = pd.read_csv(infile)
    df = df.loc[df['band'] != 'G']
    df = df.loc[~(df['rejected_by_photometry'] | df['rejected_by_variability'])]
    df['time'] = df['time'].astype(np.float32)
    df['time_resampled'] = df['time'].apply(lambda x: np.round(x, 2))
    df['interval'] = pd.cut(df['time_resampled'], 200, precision=2)
    # interval = df.groupby('interval')['source_id'].apply(
    #     lambda x: len(x.unique())).sort_values(ascending=False).head(1).index[0]
    interval = df.groupby('interval')['source_id'].count().sort_values(ascending=False).head(1).index[0]
    df = df.loc[df['interval'] == interval]
    # df['scaled'] = df.groupby(['source_id', 'band'])['flux_over_error'].transform(lambda x: x/x.max())
    # df['scaled'] = df.groupby(['source_id'])['flux_over_error'].transform(lambda x: x/x.max())
    df['scaled'] = df.groupby(['source_id', 'band'])['flux_over_error'].transform(lambda x: np.log10(1+x)-1.5)
    counts = df.groupby('source_id')['scaled'].count()
    keep = counts[counts > min_count]
    df = df.loc[df['source_id'].isin(keep.index)]
    return df

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