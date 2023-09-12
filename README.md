# Variational-Autoencoder-VAE-for-Image-Clustering


This repository contains code for performing image clustering using a Variational Autoencoder (VAE). The VAE is trained on a dataset of images, and then spectral clustering is applied to the learned embeddings to group similar images into clusters. Here is an overview of the code:

## Data Preparation
- Image files are loaded from the specified directory and resized to 64x64 pixels.
- Images are normalized to values between 0 and 1.

## VAE Architecture
- The VAE is designed with an encoder and a decoder.
- The encoder takes input images and maps them to a lower-dimensional latent space.
- The decoder reconstructs the input images from the latent space representations.
- The VAE is trained to minimize a loss function that includes both a reconstruction loss and a KL divergence term.

## Spectral Clustering
- The embeddings (latent space representations) obtained from the encoder are standardized.
- Spectral clustering is applied to the standardized embeddings to group images into clusters.
- The number of clusters is specified as `n_clusters` (in this code, it's set to 6).

## Usage
- Place your image files in the specified directory (variable `image_dir`).
- Adjust hyperparameters, such as the number of epochs for VAE training and the number of clusters for spectral clustering, as needed.
- Run the code to train the VAE and perform spectral clustering on the embeddings.
- Cluster labels for the images will be stored in the `cluster_labels` variable.

Please note that this code serves as a starting point for image clustering using VAE and spectral clustering. You may need to customize it further based on your specific dataset and requirements.

Feel free to reach out if you have any questions or need assistance with adapting this code for your project.
