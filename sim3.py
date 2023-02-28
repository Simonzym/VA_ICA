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
#read data and split into training and test set
def get_data(file_train, file_test):
       
    file_tr = pd.read_csv(file_train, index_col = 0)
    file_te = pd.read_csv(file_test, index_col = 0)
    
    train_sets = []
    test_sets = []
    for i in range(1200):
        col_name = ''.join(['V', str(i+1)])
        train_sets.append(file_tr[col_name].tolist())
        test_sets.append(file_te[col_name].tolist())
    
    return np.array(train_sets).astype('float32'), np.array(test_sets).astype('float32')


training, test = get_data('SimData/train3.csv', 'SimData/test3.csv')
train_imgs = (tf.data.Dataset.from_tensor_slices(training)
                 .shuffle(1200).batch(1))
test_imgs = (tf.data.Dataset.from_tensor_slices(test)
                 .shuffle(1200).batch(1))

epochs = 30
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 1000
#build encoder and decoder
class CVAE(tf.keras.Model):
  """MLP variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    #dimension of latent space
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000)),
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
            tf.keras.layers.Dense(1000*2, activation='tanh'),
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
  
  

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.

#train model
model = CVAE(latent_dim)

for epoch in range(1, epochs + 1):
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


#use fastICA to extract mixing matrix
from sklearn.decomposition import FastICA     
va_transformer = FastICA(n_components = 3, random_state = 0)
va_features = va_transformer.fit_transform(mean.numpy().T).T

raw_transformer = FastICA(n_components = 3, random_state = 0)
raw_features = raw_transformer.fit_transform(test.T)
