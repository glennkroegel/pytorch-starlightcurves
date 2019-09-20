"""
Created by: Glenn Kroegel
Date: 21 July 2019

Description: Training loop.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchnlp.nn import Attention
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

from autoencoder import Model

def main():
    model = Model()
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.encoder_optimizer, T_max=10, eta_min=1e-3)
    for epoch in tqdm(range(self.epochs)):
        lr = self.scheduler.get_lr(epoch)#[0]
        epoch_loss = self.train()
        # cross validation
        total_val_loss = 0
        total_dist_true = 0
        total_dist_false = 0
        with torch.no_grad():
            for _, data in enumerate(self.cv_loader):
                s1, s2, targets = data
                self.model.eval()
                val_outputs = self.model(s1, s2)
                v1, v2 = val_outputs
                val_loss = self.criterion(v1, v2, targets)
                dist_true, dist_false = euc_acc(val_outputs, targets)
                total_val_loss += val_loss.item()
                total_dist_true += dist_true.item()
                total_dist_false += dist_false.item()
            epoch_val_loss = total_val_loss/len(self.cv_loader)
            if epoch % 1 == 0:
                self.status(epoch, epoch_loss, epoch_val_loss, epoch_dist_true, epoch_dist_false, lr)
            if epoch_val_loss < self.best_loss:
                print('dumping model...')
                path = 'model' + '.pt'
                torch.save(self.model, path)
                self.best_loss = epoch_val_loss

    def status(self, epoch, epoch_loss, epoch_val_loss, epoch_dist_true, epoch_dist_false, lr):
        print('epoch {0}/{1}:\n train_loss: {2} val_loss: {3} dist_true: {4} dist_false: {5} learning_rate: {6}'
        .format(epoch, self.epochs, epoch_loss, epoch_val_loss, epoch_dist_true, epoch_dist_false, lr))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Stopping')