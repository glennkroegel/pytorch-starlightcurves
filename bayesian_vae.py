'''
created_by: Glenn Kroegel
date: 2 August 2019

'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
import shutil
from utils import count_parameters, accuracy
from config import NUM_EPOCHS

import pyro
import pyro.distributions as dist
from pyro.distributions import Normal, Categorical, Bernoulli
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam, ClippedAdam, SGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss']

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.act = nn.LeakyReLU()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.act(self.fc(x))
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias, padding=padding)
        self.act = nn.LeakyReLU()
        self.in_size = in_c
        self.out_size = out_c

    def forward(self, x):
        x = self.act(self.conv(x))
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
        self.c1 = ConvResBlock(1, 3)
        self.pool = nn.AdaptiveMaxPool1d(10)
        self.fc = Dense(30, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        bs = x.size(0)
        x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        x = self.out(x)
        return x

class Parameters:
    def __init__(self):
        self.input_dim = 1
        self.hidden_dim = 20
        self.nl = 1
        self.bidir = False
        self.direction = 1
        if self.bidir:
            self.direction = 2
        self.sl = 1024
        self.z_dim = 20

class RNNEncoder(nn.Module):
    def __init__(self, params):
        super(RNNEncoder, self).__init__()
        self.params = params
        self.rnn = nn.GRU(input_size=params.input_dim, 
                          bidirectional=params.bidir, 
                          hidden_size=params.hidden_dim, 
                          num_layers=params.nl, 
                          bias=False)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.params.nl*self.params.direction, batch_size, self.params.hidden_dim)
        return h0.to(device)

    def forward(self, x):
        bs = x.size(0)
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2) # sl,bs,xs
        h0 = self.init_hidden(bs)
        outp, hidden = self.rnn(x, h0)
        x = hidden.view(bs, -1)
        return x

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.rnn_enc = RNNEncoder(params)

        self.rnn_enc.eval()
        x = torch.randn(1, 1024)
        x.requires_grad_(False)
        x = self.rnn_enc(x)
        in_sz = x.size(1) 
        self.rnn_enc.train()

        self.fc_scale = nn.Linear(in_sz, params.z_dim)
        self.fc_loc = nn.Linear(in_sz, params.z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.rnn_enc(x)
        z_loc = self.fc_loc(x)
        z_scale = torch.exp(self.fc_scale(x))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, params):
        self.fc = nn.Linear(params.z_dim, params.hidden_dim)
        self.rnn = nn.GRU(input_size=params.hidden_dim, 
                          bidirectional=params.bidir, 
                          hidden_size=params.input_size, 
                          num_layers=params.nl, 
                          bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        pass

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.params = Parameters()
        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params)

        if torch.cuda.is_available():
            self.cuda()

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.params.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.params.z_dim)))
            z = pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
            res = self.decoder(z)
            pyro.sample("obs", Bernoulli(res).to_event(1), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))

#######################################################################################

def normal(*shape):
    loc = torch.zeros(*shape).to(device)
    scale = torch.ones(*shape).to(device)
    dist = Normal(loc, scale)#.independent(1)
    return dist

def variable_normal(name, *shape):
    l = torch.empty(*shape, requires_grad=True, device=device)
    s = torch.empty(*shape, requires_grad=True, device=device)
    torch.nn.init.normal_(l, mean=0, std=0.01)
    torch.nn.init.normal_(s, mean=0, std=0.01)
    loc = pyro.param(name+"_loc", l)
    scale = F.softplus(pyro.param(name+"_scale", s))
    return Normal(loc, scale)

#######################################################################################

def status(epoch, train_props, cv_props):
    s0 = 'epoch {0}/{1}\n'.format(epoch, NUM_EPOCHS)
    s1, s2 = '',''
    for k,v in train_props.items():
        s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
    for k,v in cv_props.items():
        s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
    print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        pyro.clear_param_store()
        clf = Classifier()
        opt = ClippedAdam({"lr": 0.01, "clip_norm": 0.01})
        svi = SVI(model=clf.model, guide=clf.guide, optim=opt, loss=Trace_ELBO())
        
        train_loader = torch.load('train_loader.pt')
        cv_loader = torch.load('cv_loader.pt')
        epochs = NUM_EPOCHS
        num_iter = 5
        best_loss = 1e50
        for epoch in tqdm(range(epochs)):
            train_props = {k:0 for k in status_properties}
            for i, data in enumerate(train_loader):
                clf.train()
                x, targets = data
                x = x.to(device)
                targets = targets.to(device)
                targets = targets.view(-1)
                loss = svi.step(x, targets)
                train_props['loss'] += loss
                preds = clf.predict(x)
                a = (preds == targets).float().mean()#accuracy(preds, targets)
                # a, a1, a2, a3 = accuracy(preds, targets)
                train_props['accuracy'] += a.item()
                # train_props['accuracy_1'] += a1
                # train_props['accuracy_2'] += a2
                # train_props['accuracy_3'] += a3
            L = len(train_loader)
            train_props = {k:v/L for k,v in train_props.items()}

            cv_props = {k:0 for k in status_properties}
            for j, data in enumerate(cv_loader):
                x, targets = data
                targets = targets.view(-1)
                x = x.to(device)
                targets = targets.to(device)
                clf.eval()
                preds = clf.predict(x)
                cv_props['loss'] += svi.evaluate_loss(x, targets)
                # preds = F.log_softmax(preds, dim=1)
                # preds = torch.argmax(preds, dim=1)
                # a, a1, a2, a3 = accuracy(preds, targets)
                # a = accuracy(preds, targets)
                a = (preds == targets).float().mean()
                cv_props['accuracy'] += a.item()
                # cv_props['accuracy_1'] += a1
                # cv_props['accuracy_2'] += a2
                # cv_props['accuracy_3'] += a3
            L = len(cv_loader)
            cv_props = {k:v/L for k,v in cv_props.items()}
            if cv_props['loss'] < best_loss:
                print('Saving state')
                state = {'state_dict': clf.state_dict(), 'train_props': train_props, 'cv_props': cv_props}
                torch.save(state, 'nn_state.pth.tar')
                torch.save(opt, 'nn_opt.pth.tar')
                best_loss = cv_props['loss']
            status(epoch, train_props, cv_props)
    except KeyboardInterrupt:
        # pd.to_pickle(mdl.train_loss, 'train_loss.pkl')
        # pd.to_pickle(mdl.cv_loss, 'cv_loss.pkl')
        print('Stopping')