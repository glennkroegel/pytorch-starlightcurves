'''
created_by: Glenn Kroegel
date: 20 July 2019

status: WIP

'''

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

from dataset import TSDataset

# http://chandlerzuo.github.io/blog/2017/11/darnn

class Conv(nn.Module):
    """input shape: (bs, features, sl)"""
    def __init__(self, in_c=1, out_c=50, ks=10, stride=1, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        return x

class Encoder(nn.Module):
    '''decs'''
    def __init__(self, rnn_input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.nl = 1
        self.rnn_input_dim = rnn_input_dim
        self.hidden_dim = hidden_dim
        self.bidir = False
        self.direction = 1
        if self.bidir:
            self.direction = 2

        self.in_c = 1
        self.out_c = 30
        self.ks = 10

        self.conv = Conv(in_c=self.in_c, out_c=self.out_c, ks=self.ks)
        self.pool = nn.AdaptiveMaxPool1d(20)
        self.rnn = nn.LSTM(input_size=self.out_c, hidden_size=self.hidden_dim, num_layers=self.nl, bidirectional=self.bidir) #(sl, bs, inp_sz)

    def init_hidden(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        h0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        c0 = Variable(torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim))#.to(self.device)
        return h0, c0

    def forward(self, x, state):
        bs = x.size(0)
        sl = x.size(1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        bs, inp_sz, sl = x.size()
        x = x.view(sl, bs, inp_sz)
        hidden, cell = state
        output, state = self.rnn(x, (hidden, cell))
        return output, state

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, rnn_input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.rnn_input_dim = rnn_input_dim
        self.hidden_dim = hidden_dim

        self.rnn_decoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.rnn_input_dim, num_layers=1, bidirectional=False)
        self.deconv = nn.ConvTranspose1d(in_channels=30, out_channels=1, kernel_size=10)

    def forward(self, input, hidden):
        bs = input.size(0)
        sl = input.size(1)
        import pdb; pdb.set_trace()
        output, (hidden, cell) = self.rnn_decoder(input, hidden)
        output = self.deconv(output)
        return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn_input_dim = 20
        self.hidden_dim = 32

        self.encoder = Encoder(self.rnn_input_dim, self.hidden_dim)
        self.decoder = Decoder(self.rnn_input_dim, self.hidden_dim)

        self.criterion = nn.MSELoss()
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-2, weight_decay=1e-6)
        self.decoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-2, weight_decay=1e-6)

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.best_loss = 1e3
        self.epochs = 10
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=10, eta_min=1e-3)

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        total_loss = 0
        for _, data in enumerate(self.train_loader):
            bs = data.size(0)
            state = self.encoder.init_hidden(bs)
            output, state = self.encoder(data, state)
            output = self.decoder(output, state)
            loss = self.criterion(output, data)
            total_loss += loss.item()
            
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        total_loss = total_loss/len(self.train_loader)
        return total_loss

    def loop(self):
        for epoch in tqdm(range(self.epochs)):
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()
            epoch_loss = self.train()
            # cross validation
            total_val_loss = 0
            with torch.no_grad():
                for _, data in enumerate(self.cv_loader):
                    bs = data.size(0)
                    state = self.encoder.init_hidden(bs)
                    output, state = self.encoder(data)
                    output = self.decoder(output, state)
                    val_loss = self.criterion(output, data)
                    total_loss += val_loss.item()
                    total_val_loss += val_loss.item()
                epoch_val_loss = total_val_loss/len(self.cv_loader)
                if epoch % 1 == 0:
                    self.status(epoch, epoch_loss, epoch_val_loss, lr)
                if epoch_val_loss < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = epoch_val_loss

        def status(self, epoch, epoch_loss, epoch_val_loss, lr):
            print('epoch {0}/{1}:\n train_loss: {2} val_loss: {3} learning_rate: {4}'
            .format(epoch, self.epochs, epoch_loss, epoch_val_loss, lr))

if __name__ == "__main__":
    try:
        mdl = Model()
        mdl.loop()
    except KeyboardInterrupt:
        print('Stopping')