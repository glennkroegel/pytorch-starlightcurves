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
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam, ClippedAdam, SGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss', 'accuracy', 'accuracy_1', 'accuracy_2', 'accuracy_3']

# https://alsibahi.xyz/snippets/2019/06/15/pyro_mnist_bnn_kl.html
# https://forum.pyro.ai/t/mini-batch-training-of-svi-models/895/8
# https://github.com/paraschopra/bayesian-neural-network-mnist/blob/master/bnn.ipynb
# checkpointing: https://pyro.ai/examples/dmm.html
# deep kernel learning - https://pyro.ai/examples/dkl.html
# https://forum.pyro.ai/t/trying-to-create-bayensian-convnets-using-pyro/563/3

#############################################################################################################################

insize = (1, 1, 28, 28)
def get_hooks(m):
    # md = {k:v for k,v in m._modules.items()}
    md = {k:v[1] for k,v in enumerate(m._modules.items())}
    hooks = {k: Hook(layer) for k, layer in md.items()}
    x = torch.randn(insize).requires_grad_(False)
    m.eval()(x)
    out_szs = {k:h.output.shape for k,h in hooks.items()}
    inp_szs = {k:h.input[0].shape for k,h in hooks.items()}
    return hooks, inp_szs, out_szs

#############################################################################################################################

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
        return torch.norm(x, dim=1) + self.c2(self.c1(x))

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

class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.nl = 1
        self.input_dim = 1
        self.hidden_dim = 8
        self.bidir = False
        self.direction = 1
        if self.bidir:
            self.direction = 2
        self.rnn = nn.GRU(input_size=self.input_dim, bidirectional=self.bidir, hidden_size=self.hidden_dim, num_layers=self.nl, bias=True)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.nl*self.direction, batch_size, self.hidden_dim)
        return h0.to(device)

    def forward(self, x):
        bs = x.size(0)
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2) # sl,bs,xs
        h0 = self.init_hidden(bs)
        outp, hidden = self.rnn(x, h0)
        x = hidden.view(bs, -1)
        return x

# class RNNEncoder(nn.Module):
#     def __init__(self):
#         ''' input: (batch_size, time_steps, in_size)'''
#         super(RNNEncoder, self).__init__()
#         self.dim_out = 10
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
#         self.rnn.flatten_parameters()
#         output, hidden = self.rnn(packed)
#         output, _ = pad_packed_sequence(output, batch_first=True)
#         output = output[rev_ind, :].contiguous()
#         hidden = hidden.transpose(0,1)[rev_ind, :, :].contiguous()
#         hidden = hidden.view(bs, -1)
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
        self.out = nn.Linear(self.in_feats, 3)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.out(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        encoder = RNNEncoder().to(device)
        encoder.eval()
        x = torch.randn(1, 1024)
        x.requires_grad_(False)
        x = encoder(x.to(device))
        head = FeedForward(x.size()).to(device)
        layers = [encoder.train(), head.train()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.view(bs, 3)
        return x 

class BasicEncoder(nn.Module):
    def __init__(self):
        super(BasicEncoder, self).__init__()
        self.fc1 = nn.Linear(128, 20, bias=False)
        self.fc_out = nn.Linear(20, 3, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.act(self.fc1(x))
        x = self.fc_out(x)
        # return F.log_softmax(x, dim=1)
        return x

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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

class Classifier(nn.Module):
    def __init__(self, use_cuda=False):
        super(Classifier, self).__init__()
        self.encoder = BasicEncoder()#Net()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.cuda()

    def model(self, inputs, targets):
        bs = targets.size(0)
        priors = {}
        for param, data in self.encoder.named_parameters():
            priors[param] = normal(data.shape)
        lifted_module = pyro.random_module("encoder", self.encoder, priors)
        lifted_reg_model = lifted_module()
        # with pyro.plate("data", min(bs, 50)):
        preds = lifted_reg_model(inputs)
        pyro.sample("obs", Categorical(logits=preds), obs=targets)

    def guide(self, inputs, targets):
        dists = {}
        for param, data in self.encoder.named_parameters():
            # if 'weight' in param or 'bias' in param:
            dists[param] = variable_normal(param, data.shape)
        lifted_module = pyro.random_module("encoder", self.encoder, dists)
        return lifted_module()

    # def predict(self, x, num_samples=10):
    #     sampled_models = [self.guide(None, None) for _ in range(num_samples)]
    #     preds = [model(x.to(device)) for model in sampled_models]
    #     preds = torch.stack(preds, dim=2)
    #     mean = torch.mean(preds, dim=2)
    #     return mean

    def predict(self, x, num_samples=10):
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        yhats = [model(x.to(device)).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return mean.argmax(dim=1)

    def sample(self, x, num_samples=100):
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        preds = [model(x.to(device)) for model in sampled_models]
        preds = torch.stack(preds, dim=2)
        return preds

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
        opt = ClippedAdam({"lr": 0.001, "clip_norm": 0.5})
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