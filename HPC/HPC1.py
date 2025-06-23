import glob
import os
import time
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pandas as pd
import nibabel as nib

#number of mixture normal densities
num_encoder = 1
shrink = pd.read_csv('shrink.csv')
#read data and split into training and test set
all_rid = (shrink['x']-1).tolist()

training = training[:,all_rid]
test = test[:,all_rid]

ic_size = len(all_rid)

train_imgs = (tf.data.Dataset.from_tensor_slices(training)
                 .shuffle(500).batch(1))
test_imgs = (tf.data.Dataset.from_tensor_slices(test)
                 .shuffle(500).batch(1))

#build encoder and decoder
class CVAE(tf.keras.Model):
  """MLP variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    #dimension of latent space
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(ic_size)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )
       
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            # No activation
            tf.keras.layers.Dense(ic_size*2, activation='tanh'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    mean, logvar = tf.split(self.decoder(z), num_or_size_splits=2, axis = 1)
    return mean, logvar

optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  xmean, xlogvar = model.decode(z)
  #conditional probability
  logpx_z = log_normal_pdf(x, xmean, xlogvar)
  #prior
  logpz = log_normal_pdf(z, 0., 0.)
  #posterior probability
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  
epochs = 30
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = ic_size

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.

#train model
model = CVAE(latent_dim)

for epoch in range(1, 50 + 1):
  start_time = time.time()
  for train_x in train_imgs:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_imgs:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()

  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  
  
#use the mean of normal distribution to approximate
pos = model.encoder(test)
mean, logvar = tf.split(pos, num_or_size_splits=2, axis=1)


np.savetxt("simHPC/mean2.csv", mean, delimiter=",")
np.savetxt('simHPC/test2.csv', test, delimiter = ',')

#use fastICA to extract mixing matrix
from sklearn.decomposition import FastICA     
va_transformer = FastICA(n_components = 5, random_state = 0)
va_features = va_transformer.fit_transform(mean.numpy().T)

raw_transformer = FastICA(n_components = 5, random_state = 0)
raw_features = raw_transformer.fit_transform(test.T)

#save the estimated mixing matrix
va_mixing = np.array(va_transformer.components_)
np.savetxt("simHPC/sim2va.csv", va_features, delimiter=",")

raw_mixing = np.array(raw_transformer.components_)
np.savetxt("simHPC/sim2raw.csv", raw_features, delimiter=",")


np.savetxt("simHPC/sim2vamix.csv", va_mixing, delimiter=",")
np.savetxt("simHPC/sim2rawmix.csv", raw_mixing, delimiter=",")
