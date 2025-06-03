#!/usr/bin/env python3

#### VARIATIONAL AUTOENCODERS FOR GENE SNPs DIMENSIONALITY REDUCTION ####

import numpy as np
import os
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import argparse
import pickle
import matplotlib.pyplot as plt

from utils import _print, import_data, load_gene_matrix, setup_data_loaders
from utils import train, evaluate, custom_evaluate_loss, encode_data
from utils import plot_test_elbo, plot_losses
from utils import bin_reconstructed, matrix_similarity, confusion_matrix, latent_correlations, snps_latent_correlation
from vae_classes import Decoder, Encoder, VAE

def main(args):
    
    # clear param store
    pyro.clear_param_store()   
    
    _print("Start.")
    
    # Obtain gene_matrix
    X, lookuptab, header = import_data(X_path=X_path, 
                                       lookuptab_path=lookuptab_path,
                                       header_path=header_path)
    
    # Subset feature matrix
    gene_matrix = load_gene_matrix(X, lookuptab, header, gene = args.gene)

    # Variable that stores input size (number of SNPs in gene)
    g_length = gene_matrix.shape[1]
    
    # Calculate dimensionality of hidden layer
    hidden_dim = int(round(g_length * (args.hidden_dim_percentage/100), 0))
    
    if args.f_decoder == "Bernoulli":
        # Normalize data between 0 and 1
        gene_matrix = gene_matrix*0.5
    
    # setup data loaders
    train_loader, test_loader = setup_data_loaders(train_indices_path, test_indices_path,
                                                   gene_matrix, batch_size=128, use_cuda=args.use_cuda,
                                                   save_indices=False)

    _print("Data loaded and ready. Starting VAE setup.")
    
    # setup the VAE
    vae = VAE(z_dim = args.z_dim, hidden_dim = hidden_dim, 
              gene_length = g_length, 
              f_dec = args.f_decoder,
              beta = args.beta, 
              use_cuda=args.use_cuda)
    
    # setup the optimizer
    optimizer = Adam({"lr": args.lr})
    
    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    
    _print("Starting training loop.")
    
    train_elbo = []
    test_elbo = []
    
    # to visualize the different parts of the loss function
    total_losses = []
    reconstruction_losses = []
    kl_divergences = []
    
    # training loop
    for epoch in range(args.num_epochs):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=args.use_cuda)
        train_elbo.append(-total_epoch_loss_train)
        # print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=args.use_cuda)
            test_elbo.append(-total_epoch_loss_test)
            # print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            lt, lr, lkl = custom_evaluate_loss(vae.model, vae.guide, test_loader)
        
            total_losses.append(lt)
            reconstruction_losses.append(lr)
            kl_divergences.append(lkl)    
    
    _print("Last train epoch average loss:", train_elbo[-1])
    _print("Last test epoch average loss:", test_elbo[-1])

    # test_elbo_plot = plot_test_elbo(test_elbo, args.num_epochs, args.test_frequency)
    # test_elbo_plot.savefig(test_elbo_plot_path)
    
    # test_losses_plot = plot_losses(reconstruction_losses, kl_divergences, total_losses, args.num_epochs, args.test_frequency)
    # test_losses_plot.savefig(test_losses_path)
    
    # Save objective functions
    # np.savez(saved_objective_functions_path, 
            #  train=train_elbo, 
            #  test_elbo=test_elbo, 
            #  test_rl = reconstruction_losses,
            #  test_kl = kl_divergences,
            #  test_custom_elbo = total_losses)
    
    # Save the model's state dictionary
    # torch.save(vae.state_dict(), saved_model)
    
    # Encode and decode gene_matrix
    encoded_data = encode_data(vae, gene_matrix, use_cuda= args.use_cuda)
    decoded_data = vae.reconstruct_vec(torch.tensor(gene_matrix))
    
    # Heatmap of the difference between input gene_matrix and reconstructed matrix
    diff_norm, diff_heatmap = matrix_similarity(gene_matrix, decoded_data.detach().numpy(), args)
    
    _print("Difference matrix (real - reconstructed) metric:", diff_norm)
    
    # # Save the figure
    # diff_heatmap.savefig(heatmap_path)    
    
    # # Correlations between latent variables and snps
    # z_snps_corr_heatmap_individual, z_snps_corr_heatmap_overlapped = snps_latent_correlation(encoded_data, gene_matrix)

    # # Correlations between latent variables
    # z_corr_heatmap = latent_correlations(encoded_data)

    # z_snps_corr_heatmap_individual.savefig(z_snps_corr_individual_path)
    # z_snps_corr_heatmap_overlapped.savefig(z_snps_corr_overlap_path)
    # z_corr_heatmap.savefig(z_corr_heatmap_path)

    # Confusion matrix
    binned_decoded_data = bin_reconstructed(decoded_data.detach().numpy(), args)
    _, acc0, acc1, acc2 = confusion_matrix(gene_matrix, binned_decoded_data, args)
    
    _print("0s accuracy:", acc0)
    _print("1s accuracy:", acc1)
    _print("2s accuracy:", acc2)
    
    # confusion_matrix_plot, acc0, acc1, acc2 = confusion_matrix(gene_matrix, binned_decoded_data, args)
    # confusion_matrix_plot.savefig(confusion_matrix_path)

    _print("End time.")

    # Save encoded gene_matrix
    np.savetxt(emb_dir + "embeddings_{}.txt".format(args.gene), encoded_data, delimiter="\t")
    
if __name__ == "__main__":
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='VAE for gene dimensionality reduction.')

    # Add arguments
    parser.add_argument('gene', type=str, help='Gene to subset from feature matrix.')
    parser.add_argument('z_dim', type=int, help='Dimensionality of latent space.')
    parser.add_argument('hidden_dim_percentage', type=int, help='Dimension of hidden layer with respect to input dimension (%).')    
    parser.add_argument('beta', type=float, help='Beta to regularize KL divergence loss term contribution.')
    parser.add_argument('f_decoder', type=str, help='Likelihood function of the decoder: Bernoulli/Categorical.')
    parser.add_argument('lr', type=float, help='Learning rate.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for GPU usage.')
    parser.add_argument('num_epochs', type=int, help='Number of epochs.')
    parser.add_argument('test_frequency', type=int, help='Report test diagnostics every x steps.')

    # Parse the arguments
    args = parser.parse_args()
    
    ### DIRECTORIES AND PATHS ###

    # Change working directory
    # os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
    os.chdir('/gpfs/projects/ub112')

    # Input and output directories
    # data_dir = "data/features/"
    # biomart_dir = "data/biomart/"
    # output_dir = "data/vae/"
    data_dir = "data/enroll_hd/features/"
    biomart_dir = "data/enroll_hd/biomart/"
    output_dir = "data/enroll_hd/vae/"
    emb_dir = "data/enroll_hd/vae_embeddings/"

    # Data path
    # X_path = data_dir + "X_pc10_filt_0.01.txt"
    # header_path = data_dir + "subsetting/header_X_pc10_filt_0.01.txt"

    X_path = data_dir + "feature_matrix_m3_filt_0.01.txt"
    header_path = data_dir + "header_feature_matrix_m3_filt_0.01.txt"

    lookuptab_path = biomart_dir + "revised_filtered_snp_gene_lookup_tab.txt"
    
    output_sufix = "_{}_fdec{}_zdim{}_hdimprop{}_beta{}".format(args.gene, args.f_decoder,args.z_dim,args.hidden_dim_percentage, args.beta)
    
    saved_objective_functions_path = output_dir + "objective_functions" + output_sufix
    saved_model = output_dir + "model" + output_sufix
    test_elbo_plot_path = output_dir + "test_elbo" + output_sufix + ".png"
    test_losses_path = output_dir + "losses" + output_sufix + ".png"
    heatmap_path = output_dir + "diff_heatmap" + output_sufix + ".png"
    z_snps_corr_individual_path = output_dir + "z_snps_corr_individual" + output_sufix + ".png"
    z_snps_corr_overlap_path = output_dir + "z_snps_corr_overlap" + output_sufix + ".png"
    z_corr_heatmap_path = output_dir + "z_corr_heatmap" + output_sufix + ".png"
    confusion_matrix_path = output_dir + "confusion_matrix" + output_sufix + ".png"
    train_indices_path = output_dir + "train_idxs" + output_sufix + ".txt"
    test_indices_path = output_dir + "test_idxs" + output_sufix + ".txt"
    
    model = main(args)