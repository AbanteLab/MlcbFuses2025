#!/usr/bin/env python3

### Classes for VAE training ###

import torch
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist

# Decoder network
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, gene_length, f_dec):
        super().__init__()
        self.f_dec = f_dec
        self.gene_length = gene_length
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        if f_dec == "Bernoulli":
            self.fc21 = nn.Linear(hidden_dim, gene_length)
        elif f_dec == "Categorical":
            self.fc21 = nn.Linear(hidden_dim, gene_length * 3)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x number of SNPs in gene (gene_length)
        if self.f_dec == "Bernoulli":
            loc_img = self.sigmoid(self.fc21(hidden))
            return loc_img
        elif self.f_dec == "Categorical":
            logits = self.fc21(hidden).view(-1, self.gene_length, 3)
            return logits
        
# Encoder network
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, gene_length):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(gene_length, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the vector x
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale
    
class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, gene_length, f_dec, beta = 1, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, gene_length)
        self.decoder = Decoder(z_dim, hidden_dim, gene_length, f_dec)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.gene_length = gene_length
        self.f_dec = f_dec
        self.beta = beta

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            with pyro.poutine.scale(scale=self.beta):
                # sample from prior (value will be sampled by guide when computing the ELBO)
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1)) # multivariate normal as joint distribution
            # decode the latent code z
            dec_out = self.decoder(z)
            # likelihood of x given the reconstructed data
            # score against actual images
            if self.f_dec == "Bernoulli":
                pyro.sample("obs", dist.Bernoulli(probs = dec_out, validate_args=False).to_event(1), obs=x)
            elif self.f_dec == "Categorical":
                pyro.sample("obs", dist.Categorical(logits = dec_out).to_event(1), obs=x)
        
        return dec_out

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=self.beta):
                # use the encoder to get the parameters used to define q(z|x)
                z_loc, z_scale = self.encoder(x)
                # sample the latent code z
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
        return z_loc, z_scale

    # define a helper function for reconstructing vectors
    def reconstruct_vec(self, x):
        # encode x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode 
        loc_vec = self.decoder(z)
        return loc_vec