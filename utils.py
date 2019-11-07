import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os

def T(a):
    if torch.is_tensor(a):
        res = a
    else:
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            res = torch.FloatTensor(a.astype(np.float32))
        else:
            raise NotImplementedError(a.dtype)
    return to_gpu(res)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

USE_GPU=False
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def accuracy(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    inp = input.argmax(dim=-1).view(-1)
    targs = targs.view(n,-1)
    # ix1 = (targs == 0).nonzero().squeeze()
    # ix2 = (targs == 1).nonzero().squeeze()
    # ix3 = (targs == 2).nonzero().squeeze()
    # acc1 = (inp[ix1] == targs[ix1]).float().mean().item()
    # acc2 = (inp[ix2] == targs[ix2]).float().mean().item()
    # acc3 = (inp[ix3] == targs[ix3]).float().mean().item()
    acc = (inp==targs).float().mean()
    return acc#, acc1, acc2, acc3

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

def bce_acc_superseded(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    bs = input.size(0)
    input = torch.round(input).view(bs,1)
    targs = targs.view(bs,1)
    return (input==targs).float().mean()

def cos_acc(input, targs):
    "Computes correct similarity classification for two embeddings."
    v1, v2 = input
    bs = v1.size(0)
    p = F.cosine_similarity(v1,v2)
    p[p < 0] = 0
    preds = torch.round(p).view(bs,1)
    targs = targs.view(bs,1)
    targs[targs==-1] = 0
    acc = (preds==targs).float().mean()

    # False positive rate in high threshold (hard negatives)
    ixs = (p > 0.8).nonzero().squeeze()
    strong_pairs = torch.round(p[ixs])
    strong_targs = targs[ixs]
    hard_fp_rate = (strong_pairs != strong_targs).float().mean()
    return acc, hard_fp_rate

def euc_acc(input, targs):
    "Running statistics for euclidean distance models"
    v1, v2 = input
    bs = v1.size(0)
    d = torch.abs(v1-v2).sum(dim=1)
    ixs_true = (targs == 1).nonzero().squeeze()
    ixs_false = (targs == 0).nonzero().squeeze()
    dist_true = d[ixs_true].mean()
    dist_false = d[ixs_false].mean()
    return dist_true, dist_false

def recall(input):
    pass

def triple_accuracy(vecs):
    v1, v2, v3 = vecs
    bs = v1.size(0)
    ap_sim = F.cosine_similarity(v1, v2)
    an_sim = F.cosine_similarity(v1, v3)
    ap_mean = ap_sim.mean()
    ap_min = ap_sim.min()
    an_mean = an_sim.mean()
    return ap_min, ap_mean, an_mean

def fbeta(y_pred, y_true, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `y_pred` and `y_true` in a multi-classification task."
    beta2 = beta**2
    import pdb; pdb.set_trace()
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()

###############################################################################################

def check_if_numbers_are_consecutive(list_):
    """
    https://github.com/amirziai/flatten
    """
    return all((True if second - first == 1 else False
                for first, second in zip(list_[:-1], list_[1:])))

def _construct_key(previous_key, separator, new_key):
    """
    Returns the new_key if no previous key exists, otherwise concatenates
    previous key, separator, and new_key
    :param previous_key:
    :param separator:
    :param new_key:
    :return: a string if previous_key exists and simply passes through the
    new_key otherwise
    """
    if previous_key:
        return u"{}{}{}".format(previous_key, separator, new_key)
    else:
        return new_key

def flatten(nested_dict, separator="_", root_keys_to_ignore=set()):
    """
    Flattens a dictionary with nested structure to a dictionary with no
    hierarchy
    Consider ignoring keys that you are not interested in to prevent
    unnecessary processing
    This is specially true for very deep objects

    :param nested_dict: dictionary we want to flatten
    :param separator: string to separate dictionary keys by
    :param root_keys_to_ignore: set of root keys to ignore from flattening
    :return: flattened dictionary
    """
    assert isinstance(nested_dict, dict), "flatten requires a dictionary input"
    assert isinstance(separator, six.string_types), "separator must be string"

    # This global dictionary stores the flattened keys and values and is
    # ultimately returned
    flattened_dict = dict()

    def _flatten(object_, key):
        """
        For dict, list and set objects_ calls itself on the elements and for
        other types assigns the object_ to
        the corresponding key in the global flattened_dict
        :param object_: object to flatten
        :param key: carries the concatenated key for the object_
        :return: None
        """
        # Empty object can't be iterated, take as is
        if not object_:
            flattened_dict[key] = object_
        # These object types support iteration
        elif isinstance(object_, dict):
            for object_key in object_:
                if not (not key and object_key in root_keys_to_ignore):
                    _flatten(object_[object_key], _construct_key(key,
                                                                 separator,
                                                                 object_key))
        elif isinstance(object_, (list, set, tuple)):
            for index, item in enumerate(object_):
                _flatten(item, _construct_key(key, separator, index))
        # Anything left take as is
        else:
            flattened_dict[key] = object_

    _flatten(nested_dict, None)
    return flattened_dict

    ###############################################################################################

def binary_search_last_index_less_or_equal(sorted_in_array, value, start_index=0, end_index=-1, method_get_key=None):
    if start_index == end_index:
        return start_index
    if end_index == -1:
        end_index = len(sorted_in_array) - 1
    if start_index == -1:
        start_index = len(sorted_in_array) - 1
    center_index = math.ceil(start_index + (end_index - start_index) / 2)
    center_value = sorted_in_array[center_index]
    if method_get_key:
        center_value = method_get_key(center_value)
    if center_value > value:
        return binary_search_last_index_less_or_equal(sorted_in_array, value, start_index=start_index, end_index=center_index - 1, method_get_key=method_get_key)
    else:
        return binary_search_last_index_less_or_equal(sorted_in_array, value, start_index=center_index, end_index=end_index, method_get_key=method_get_key)