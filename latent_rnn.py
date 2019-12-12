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
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################
# Options
input_dim = 1
classif_per_tp = False
n_labels = 1
obsrv_std = 0.01
niters = 1
status_properties = ['loss']
latent_dim = 40

##################################################################

def create_LatentODE_model(input_dim, z0_prior, obsrv_std, device = device, classif_per_tp = False, n_labels = 1, latents=latent_dim):

    dim = latent_dim
    ode_func_net = utils.create_net(dim, latents, 
        n_layers = 1, n_units = 20, nonlinear = nn.Tanh)

    gen_ode_func = ODEFunc(
        input_dim = input_dim, 
        latent_dim = latent_dim, 
        ode_func_net = ode_func_net,
        device = device).to(device)
        
    z0_diffeq_solver = None
    n_rec_dims = 20 # rec_dims: default 20
    enc_input_dim = int(input_dim) * 2 # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = latent_dim

    ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
        n_layers = 1, n_units = 20, nonlinear = nn.Tanh)

    rec_ode_func = ODEFunc(
        input_dim = enc_input_dim, 
        latent_dim = n_rec_dims,
        ode_func_net = ode_func_net,
        device = device).to(device)

    z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", latents, 
        odeint_rtol = 1e-5, odeint_atol = 1e-6, device = device)

    encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
        z0_dim = z0_dim, n_gru_units = 20, device = device).to(device)

    decoder = Decoder(latents, input_dim=1).to(device)

    diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', latents, 
        odeint_rtol = 1e-5, odeint_atol = 1e-6, device = device)

    model = LatentODE(
        input_dim = gen_data_dim, 
        latent_dim = latents, 
        encoder_z0 = encoder_z0, 
        decoder = decoder, 
        diffeq_solver = diffeq_solver, 
        z0_prior = z0_prior, 
        device = device,
        obsrv_std = obsrv_std,
        use_poisson_proc = False, 
        use_binary_classif = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = n_labels,
        train_classif_w_reconstr = False
        ).to(device)

    return model

##################################################################
# Model
obsrv_std = torch.Tensor([0.01]).to(device)
z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

model = create_LatentODE_model(input_dim, z0_prior, obsrv_std)

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
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.1)
    # train_loader = torch.load('vae_train_loader.pt')
    # test_loader = torch.load('vae_cv_loader.pt')
    train_loader = torch.load('toy_train.pt')
    test_loader = torch.load('toy_cv.pt')
    num_batches = len(train_loader)
    kl_wait = 1
    num_epochs = NUM_EPOCHS
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs)):
        kl_coef = (1 - 0.9 ** epoch) if epoch > kl_wait else 0
        print(kl_coef)
        train_loss = 0
        train_props = {k:0 for k in status_properties}
        for i, data in enumerate(train_loader):
            if i % 20 == 0:
                print(i)
            optimizer.zero_grad()
            train_res = model.compute_all_losses(data, n_traj_samples=20, kl_coef=kl_coef)
            train_res['loss'].backward()
            train_props['loss'] += train_res['loss'].detach().item()
            optimizer.step()
        train_props = {k:v/len(train_loader) for k,v in train_props.items()}
        loss = train_props['loss']
        status(epoch, train_props)
        if loss < best_loss:
            best_loss = loss
            print('Saving state...')
            torch.save({'epoch': epoch, 'loss': loss, 'state_dict': model.state_dict()}, 'latent_ode_state.pth.tar')


        

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