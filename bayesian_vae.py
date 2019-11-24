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

# https://github.com/kefirski/pytorch_RVAE/blob/master/model/decoder.py

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
        self.c1 = ConvResBlock(1, 10)
        self.pool = nn.AdaptiveMaxPool1d(10)
        self.fc = Dense(100, 10)
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
        self.hidden_dim = 12
        self.nl = 2
        self.bidir = False
        self.direction = 1
        if self.bidir:
            self.direction = 2
        self.sl = 128
        self.z_dim = 20
        self.decoder_hidden = 8
        self.rnn_out_sz = self.hidden_dim * self.nl * self.direction

class RNNEncoder(nn.Module):
    def __init__(self, params):
        super(RNNEncoder, self).__init__()
        self.params = params
        self.rnn = nn.GRU(input_size=params.input_dim, 
                          bidirectional=params.bidir, 
                          hidden_size=params.hidden_dim, 
                          num_layers=params.nl, 
                          bias=False)
        self.fc = nn.Sequential(nn.Linear(params.rnn_out_sz, 40), nn.Tanh(), nn.Linear(40, 20))

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
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.rnn_enc = RNNEncoder(params)

        self.rnn_enc.eval()
        x = torch.randn(1, 128)
        x.requires_grad_(False)
        x = self.rnn_enc(x)
        in_sz = x.size(1) 
        self.rnn_enc.train()

        self.fc_scale = nn.Linear(in_sz, params.z_dim)
        self.fc_loc = nn.Linear(in_sz, params.z_dim)

    def forward(self, x):
        x = self.rnn_enc(x)
        z_loc = self.fc_loc(x)
        z_scale = torch.exp(self.fc_scale(x))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.decoder = nn.Sequential(nn.Linear(params.z_dim, params.sl))

    def forward(self, z):
        x = self.decoder(z)
        return x

class ConvEnc(nn.Module):
    def __init__(self, params):
        super(ConvEnc, self).__init__()
        self.c1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(5, 10, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.25)
        self.pool = nn.AdaptiveMaxPool1d(40)
        
        self.fc_scale = nn.Linear(100, params.z_dim)
        self.fc_loc = nn.Linear(100, params.z_dim)

    def forward(self, x):
        bs = x.size(0)
        x.unsqueeze_(1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.pool(x)
        x = x.view(bs, -1)
        z_loc = self.fc_loc(x)
        z_scale = torch.exp(self.fc_scale(x))
        return z_loc, z_scale

class ConvDec(nn.Module):
    def __init__(self, params):
        super(ConvDec, self).__init__()
        self.params = params
        self.c1 = nn.Conv1d(10, 5, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(5, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(params.z_dim, 100)
        self.act = nn.LeakyReLU(negative_slope=0.25)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        bs = z.size(0)
        x = self.softplus(self.fc(z))
        x = x.view(bs, 10, 10)
        x = F.interpolate(x, self.params.sl)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

class RNNDecoder(nn.Module):
    def __init__(self, params):
        super(RNNDecoder, self).__init__()
        self.params = params
        self.fc = nn.Linear(params.z_dim, params.decoder_hidden)
        self.rnn = nn.GRU(input_size=params.hidden_dim, 
                          bidirectional=params.bidir, 
                          hidden_size=params.decoder_hidden, 
                          num_layers=params.nl, 
                          bias=False)
        self.fc_out = nn.Linear(params.decoder_hidden, 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        bs = z.size(0)
        hidden = self.softplus(self.fc(z))
        hidden = hidden.unsqueeze(0)
        x = torch.zeros(1024, bs, self.params.hidden_dim)
        outp, _ = self.rnn(x, hidden)
        outp = self.fc_out(outp)
        outp = self.sigmoid(outp)
        outp = outp.squeeze().transpose(0,1)
        return outp

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

    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z = Normal(z_loc, z_scale).sample()
        res = self.decoder(z)
        return res

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
        vae = VAE()
        opt = ClippedAdam({"lr": 0.01, "clip_norm": 0.25})
        svi = SVI(model=vae.model, guide=vae.guide, optim=opt, loss=Trace_ELBO())
        
        train_loader = torch.load('vae_train_loader.pt')
        cv_loader = torch.load('vae_cv_loader.pt')
        epochs = NUM_EPOCHS
        num_iter = 5
        best_loss = 1e50
        for epoch in tqdm(range(epochs)):
            train_props = {k:0 for k in status_properties}
            for i, x in enumerate(train_loader):
                vae.train()
                x = x.to(device)
                loss = svi.step(x)
                train_props['loss'] += loss
            L = len(train_loader)
            train_props = {k:v/L for k,v in train_props.items()}

            cv_props = {k:0 for k in status_properties}
            for j, x in enumerate(cv_loader):
                x = x.to(device)
                vae.eval()
                cv_props['loss'] += svi.evaluate_loss(x)
            L = len(cv_loader)
            cv_props = {k:v/L for k,v in cv_props.items()}
            if cv_props['loss'] < best_loss:
                print('Saving state')
                state = {'state_dict': vae.state_dict(), 'train_props': train_props, 'cv_props': cv_props}
                torch.save(state, 'nn_state.pth.tar')
                torch.save(opt, 'nn_opt.pth.tar')
                best_loss = cv_props['loss']
            status(epoch, train_props, cv_props)
    except KeyboardInterrupt:
        print('Stopping')