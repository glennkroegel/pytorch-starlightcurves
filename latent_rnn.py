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
from utils import count_parameters, accuracy, bce_acc, accuracy_from_logits, one_hot
from config import NUM_EPOCHS
from losses import BCEDiceLoss
from callbacks import Hook
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##################################################################
# Training

if __name__ == '__main__':

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    train_loader = torch.load('tess_train.pt')
    num_batches = len(train_loader)

    for itr in range(1, num_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(train_loader)
        import pdb; pdb.set_trace()
        train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()
        optimizer.step()

        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * num_batches) == 0:
            with torch.no_grad():

                test_res = compute_loss_all_batches(model, 
                    data_obj["test_dataloader"], args,
                    n_batches = data_obj["n_test_batches"],
                    experimentID = experimentID,
                    device = device,
                    n_traj_samples = 3, kl_coef = kl_coef)

                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr//num_batches, 
                    test_res["loss"].detach(), test_res["likelihood"].detach(), 
                    test_res["kl_first_p"], test_res["std_first_p"])
            
                logger.info("Experiment " + str(experimentID))
                logger.info(message)
                logger.info("KL coef: {}".format(kl_coef))
                logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
                logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
                
                if "auc" in test_res:
                    logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

                if "mse" in test_res:
                    logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

                if "accuracy" in train_res:
                    logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

                if "accuracy" in test_res:
                    logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

                if "pois_likelihood" in test_res:
                    logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

                if "ce_loss" in test_res:
                    logger.info("CE loss: {}".format(test_res["ce_loss"]))

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)


            # Plotting
            if args.viz:
                with torch.no_grad():
                    test_dict = utils.get_next_batch(data_obj["test_dataloader"])

                    print("plotting....")
                    if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
                        plot_id = itr // num_batches // n_iters_to_viz
                        viz.draw_all_plots_one_dim(test_dict, model, 
                            plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
                            experimentID = experimentID, save=True)
                        plt.pause(0.01)
    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)