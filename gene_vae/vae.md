
With the goal of reducing the dimensionality of our data before approaching GNN implementation, we explore a way of summarising SNP information based on variational autoencoders.

`vae_allgenes.py` loads feature matrix for a given's gene autoencoder training. Takes as script arguments the gene name and several hyperparameters. Uses classes for decoder, encoder and VAE defined in `vae_classes.py` (`Encoder`, `Decoder` and `VAE`) and several functions contained in `utils.py` for loading data, training and testing of VAE, and result representation. 

It also generates plots showing reconstruction results and correlations between the latent variables. The autoencoder's state dictionary is saved alongside the loss functions values over epochs. The indices of the samples used in training and testing are also saved.

`z_dim_exploration.ipynb` is an exploration of the optimal dimension of the latent space, using the approach of potentially matching the latent variables with the linkage disequilibrium blocks. `automatic_zdims.py` computes the latent dimension that will be finally used in VAE training following the previous conclusions.

Similarly, `betas_summary.py` trains several VAEs with different beta values to summarize the losses as a function of beta.
