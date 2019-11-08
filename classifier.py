'''
created_by: Glenn Kroegel
date: 2 August 2019

'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
from utils import count_parameters, accuracy, bce_acc, accuracy_from_logits
from config import NUM_EPOCHS
from callbacks import Hook
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss', 'accuracy', 'accuracy_1', 'accuracy_2', 'accuracy_3']

#############################################################################################################################

insize = (1, 1024)
def get_hooks(m):
    md = {k:v[1] for k,v in enumerate(m._modules.items())}
    hooks = {k: Hook(layer) for k, layer in md.items()}
    x = torch.randn(insize).requires_grad_(False)
    m.eval()(x)
    out_szs = {k:h.output[1].shape for k,h in hooks.items()}
    # inp_szs = {k:h.input.shape for k,h in hooks.items()}
    inp_szs = None
    return hooks, inp_szs, out_szs

#############################################################################################################################

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.bn = nn.BatchNorm1d(out_size)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.bn(self.drop(self.act(self.fc(x))))
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.in_size = in_c
        self.out_size = out_c

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.c1 = Conv(n, n)
        self.c2 = Conv(n, n)

    def forward(self, x):
        return x + self.c2(self.c1(x))

class ConvResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvResBlock, self).__init__()
        self.conv = Conv(in_c=in_c, out_c=out_c, stride=2)
        self.res_block = ResBlock(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)
        return x

#############################################################################################################################

class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.c1 = ConvResBlock(1, 10)
        self.pool = nn.AdaptiveMaxPool1d(10)
        self.fc = Dense(100, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        bs = x.size(0)
        x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.pool(x)
        x = self.fc(x)
        x = self.out(x)
        return x

class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.nl = 1
        self.input_dim = 1
        self.hidden_dim = 10
        self.bidir = False
        self.direction = 1
        if self.bidir:
            self.direction = 2
        self.rnn = nn.GRU(input_size=self.input_dim, bidirectional=self.bidir, hidden_size=self.hidden_dim, num_layers=self.nl, bias=False)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim, device=device)
        return h0

    def forward(self, x):
        bs = x.size(0)
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2) # sl,bs,xs
        h0 = self.init_hidden(bs)
        _, hidden = self.rnn(x, h0)
        x = hidden.view(bs, -1)
        return x

# class RNNEncoder(nn.Module):
#     def __init__(self):
#         ''' input: (batch_size, time_steps, in_size)'''
#         super(RNNEncoder, self).__init__()
#         self.dim_out = 12
#         self.nl = 1
#         self.rnn = nn.GRU(1, self.dim_out, num_layers=self.nl, bidirectional=False, dropout=0.1, batch_first=True, bias=False)

#     def forward(self, x):
#         x = x.unsqueeze(2)
#         bs = len(x)
#         lens = [a.size(0) for a in x]
#         indices = np.argsort(lens)[::-1].tolist()
#         rev_ind = [indices.index(i) for i in range(len(indices))]
#         x = [x[i] for i in indices]
#         # x = pad_sequence([a.transpose(0,1) for a in x], batch_first=True)
#         x = pad_sequence(x, batch_first=True)
#         input_lengths = [lens[i] for i in indices]
#         packed = pack_padded_sequence(x, input_lengths, batch_first=True)
#         output, hidden = self.rnn(packed)
#         output, _ = pad_packed_sequence(output, batch_first=True)
#         output = output[rev_ind, :].contiguous()
#         hidden = hidden.transpose(0,1)[rev_ind, :, :].contiguous()
#         return hidden

class FeedForward(nn.Module):
    def __init__(self, in_shp):
        super(FeedForward, self).__init__()
        self.in_shp = in_shp
        if len(in_shp) == 3:
            self.in_feats = in_shp[1]*in_shp[2]
        else:
            self.in_feats = in_shp[1]
        # self.fc = Dense(self.in_feats, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, self.in_feats)
        # x = self.fc(x)
        x = self.out(x)
        return x

class Classifier(nn.Module):
    def __init__(self, use_cuda=False):
        super(Classifier, self).__init__()
        encoder = RNNEncoder().to(device)
        # hooks, _, enc_szs = get_hooks(encoder)
        # idxs = list(enc_szs.keys())
        # x_sz = enc_szs[len(enc_szs) - 1]
        x = torch.randn(1, 1024)
        x.requires_grad_(False)
        x = encoder(x.to(device))
        head = FeedForward(x.size()).to(device)
        layers = [encoder.to(device), head.to(device)]
        [print(count_parameters(x)) for x in layers]
        self.layers = nn.Sequential(*layers)

        if use_cuda:
            self.cuda()

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.view(bs, 3)
        return x 

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

#######################################################################################

class BaseLearner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = Classifier().to(device) #ConvolutionalEncoder()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-2, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-3)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.train_loss = []
        self.cv_loss = []

        self.best_loss = 1e3
        print('Model Parameters: ', count_parameters(self.model))

    def iterate(self, loader, model, criterion, optimizer, training=True):
        if training:
            model.train()
        else:
            model.eval()
        props = {k:0 for k in status_properties}
        for i, data in enumerate(loader):
            x, targets = data
            x = x.to(device)
            targets = targets.view(-1).to(device)
            preds = model(x)
            loss = criterion(preds, targets)
            props['loss'] += loss.item()
            a, a1, a2, a3 = accuracy_from_logits(preds, targets)
            props['accuracy'] += a.item()
            props['accuracy_1'] += a1
            props['accuracy_2'] += a2
            props['accuracy_3'] += a3
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_grad_norm_(model.parameters(), 0.5)
            L = len(loader)
        props = {k:v/L for k,v in props.items()}
        return props

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            train_props = self.iterate(self.train_loader, self.model, self.criterion, self.optimizer, training=True)
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            self.train_loss.append(train_props['loss'])
            # cross validation
            with torch.no_grad():
                cv_props = self.iterate(self.cv_loader, self.model, self.criterion, self.optimizer, training=False)
                self.cv_loss.append(cv_props['loss'])
                if epoch % 1 == 0:
                    self.status(epoch, train_props, cv_props)
                if cv_props['loss'] < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = cv_props['loss']
                    is_best = True
                save_checkpoint(
                    {'epoch': epoch + 1,
                    'lr': lr, 
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'best_loss': self.best_loss}, is_best=is_best)
                is_best=False

    def status(self, epoch, train_props, cv_props):
        s0 = 'epoch {0}/{1}\n'.format(epoch, self.epochs)
        s1, s2 = '',''
        for k,v in train_props.items():
            s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        mdl = BaseLearner()
        mdl.step()
    except KeyboardInterrupt:
        print('Stopping')