'''
created_by: Glenn Kroegel
date: 1 December 2019

'''

import sys
sys.path.insert(0, '../')
sys.path.insert(1, '../latent_ode/')
import latent_ode.lib as ode
import latent_ode.lib.utils as utils
from latent_ode.lib.latent_ode import LatentODE
from latent_ode.lib.ode_rnn import ODE_RNN
from latent_ode.lib.encoder_decoder import Encoder_z0_ODE_RNN, Decoder
from latent_ode.lib.diffeq_solver import DiffeqSolver
from latent_ode.lib.ode_func import ODEFunc

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.distributions.normal import Normal
from tqdm import tqdm
from utils import count_parameters, accuracy, bce_acc, accuracy_from_logits, one_hot
from config import NUM_EPOCHS
from loading import TessDataset
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################
# Options
input_dim = 1
classif_per_tp = False
n_labels = 1
niters = 1
status_properties = ['loss', 'ratio']
latent_dim = 40

##################################################################
# Model
obsrv_std = torch.Tensor([0.1]).to(device)
z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
gru_units = 40
n_ode_gru_dims = latent_dim
				
ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
    n_layers = 2, n_units = 100, nonlinear = nn.Tanh)

rec_ode_func = ODEFunc(
    input_dim = input_dim, 
    latent_dim = n_ode_gru_dims,
    ode_func_net = ode_func_net,
    device = device).to(device)

z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "dopri5", latent_dim, 
    odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

model = ODE_RNN(input_dim=input_dim, latent_dim=latent_dim, 
            n_gru_units = 100, n_units = 100, device = device, 
			z0_diffeq_solver = z0_diffeq_solver,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = False,
			classif_per_tp = False,
			n_labels = 1,
			train_classif_w_reconstr = False
			).to(device)

disable_bias = True
if disable_bias:
    for module in model.modules():
        if hasattr(module, 'bias'):
            module.bias = None

##################################################################
# Training

def status(epoch, train_props, cv_props=None):
    s0 = 'epoch {0}/{1}\n'.format(epoch, NUM_EPOCHS)
    s1, s2 = '',''
    for k,v in train_props.items():
        s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
    if cv_props:
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)
    else:
        print(s0 + s1)

if __name__ == '__main__':
    
    print(model)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-2)
    # train_loader = torch.load('vae_train_loader.pt')
    # test_loader = torch.load('vae_cv_loader.pt')
    # train_loader = torch.load('toy_train.pt')
    train_loader = torch.load('tess_train.pt')
    cv_loader = torch.load('tess_cv.pt')
    # test_loader = torch.load('toy_cv.pt')
    num_batches = len(train_loader)
    kl_wait = 5
    num_epochs = NUM_EPOCHS
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs)):
        # kl_coef = (1 - 0.95 ** (epoch - kl_wait)) if epoch > kl_wait else 0
        # print(kl_coef)
        train_loss = 0
        train_props = {k:0 for k in status_properties}
        for i, data in enumerate(train_loader):
            if i % 20 == 0:
                print(i)
            optimizer.zero_grad()
            train_res = model.compute_all_losses(data, n_traj_samples=50)
            train_res['loss'].backward()
            train_props['loss'] += train_res['loss'].item()
            train_props['ratio'] += (train_res["kl_first_p"] / train_res["mse"]).item()
            optimizer.step()
        train_props = {k:v/len(train_loader) for k,v in train_props.items()}
        loss = train_props['loss']
        cv_props = {k:0 for k in status_properties}
        with torch.no_grad():
            for i, data in enumerate(cv_loader):
                cv_res = model.compute_all_losses(data, n_traj_samples=50)
                cv_props['loss'] += cv_res['loss'].detach().item()
                cv_props['ratio'] += (cv_res["kl_first_p"] / cv_res["mse"]).item()
            cv_props = {k:v/len(cv_loader) for k,v in cv_props.items()}
        cv_loss = cv_props['loss']
        status(epoch, train_props, cv_props)
        if cv_loss < best_loss:
            best_loss = cv_loss
            print('Saving state...')
            torch.save({'epoch': epoch, 'loss': loss, 'state_dict': model.state_dict()}, 'ode_rnn_state_tess.pth.tar')
        if epoch > 50:
            torch.save({'epoch': epoch, 'loss': loss, 'state_dict': model.state_dict()}, 'ode_rnn_state_tess50.pth.tar')


        

    # for itr in range(1, num_batches * (args.niters + 1)):
    #     optimizer.zero_grad()
    #     ode.utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

    #     wait_until_kl_inc = 10
    #     if itr // num_batches < wait_until_kl_inc:
    #         kl_coef = 0.
    #     else:
    #         kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

    #     batch_dict = next(train_loader)
    #     import pdb; pdb.set_trace()
    #     train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
    #     train_res["loss"].backward()
    #     optimizer.step()

    #     n_iters_to_viz = 1
    #     if itr % (n_iters_to_viz * num_batches) == 0:
    #         with torch.no_grad():

    #             test_res = compute_loss_all_batches(model, 
    #                 data_obj["test_dataloader"], args,
    #                 n_batches = data_obj["n_test_batches"],
    #                 experimentID = experimentID,
    #                 device = device,
    #                 n_traj_samples = 3, kl_coef = kl_coef)

    #             message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
    #                 itr//num_batches, 
    #                 test_res["loss"].detach(), test_res["likelihood"].detach(), 
    #                 test_res["kl_first_p"], test_res["std_first_p"])

    #         torch.save({
    #             'loss': loss,
    #             'state_dict': model.state_dict(),
    #         }, 'latent_ode_checkpoint.pth.tar')