# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:35:30 2023

@author: HP}
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image



# Set the path to the directory containing your images
image_dir = "data"



# List all files in the directory
image_files = os.listdir(image_dir)



# Initialize a list to store the loaded images
images = []



# Loop through each image file
for file_name in image_files:
    if file_name.endswith(".JPG"):
        # Load the image using PIL and resize
        image_path = os.path.join(image_dir, file_name)
        image = Image.open(image_path)
        image = image.resize((64, 64))  # Resize to desired dimensions
        images.append(np.array(image))
        

# Convert the list of images to a NumPy array
X_train = np.array(images)



X_train = X_train.astype('float32') / 255.0


# Define the architecture of the Variational Autoencoder (VAE)
input_shape = X_train[0].shape
latent_dim = 64

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# Encoder
input_layer = Input(shape=input_shape)
flat_input = Flatten()(input_layer)
encoded = Dense(512, activation='relu')(flat_input)
z_mean = Dense(latent_dim)(encoded)
z_log_var = Dense(latent_dim)(encoded)


def sampling(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon



z = Lambda(sampling)([z_mean, z_log_var])


# Decoder
decoded = Dense(512, activation='relu')(z)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
decoded = Reshape(input_shape)(decoded)



# VAE model
vae = Model(inputs=input_layer, outputs=decoded)




# Define VAE loss function
def vae_loss(x, decoded):
    reconstruction_loss = tf.reduce_mean(tf.square(x - decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss



vae.compile(optimizer='adam', loss=vae_loss)



# Train the VAE
vae.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True)


# Extract embeddings (latent space representations) from the encoder
encoder = Model(inputs=input_layer, outputs=z_mean)
embeddings = encoder.predict(X_train)



# Perform spectral clustering on embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)
n_clusters = 6
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
cluster_labels = spectral_clustering.fit_predict(scaled_embeddings)
