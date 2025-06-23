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
from tensorflow.keras.regularizers import l2

#number of mixture normal densities
num_encoder = 5
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


training, test = get_data('SimData/train31.csv', 'SimData/test31.csv')
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

    self.encoder1 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer = l2(0.01)),
        ]
    )
    self.encoder2 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer = l2(0.01)),
        ]
    )
    self.encoder3 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer = l2(0.01)),
        ]
    )
    self.encoder4 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer = l2(0.01)),
        ]
    )
      
    self.encoder5 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer = l2(0.01)),
        ]
    )

    
    self.coeff = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(1000,)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer = l2(0.01)),
            # No activation for the output layer
            #assume that the covariance matrix is diagonal
            #the output is mean and std of each element
            tf.keras.layers.Dense(5, activation = 'softmax'),
        ]
    )
    
    
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer = l2(0.01)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer = l2(0.01)),
            # No activation
            tf.keras.layers.Dense(1000*2, activation='tanh', kernel_regularizer = l2(0.01)),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean1, logvar1 = tf.split(self.encoder1(x), num_or_size_splits=2, axis=1)
    mean2, logvar2 = tf.split(self.encoder2(x), num_or_size_splits=2, axis=1)
    mean3, logvar3 = tf.split(self.encoder3(x), num_or_size_splits=2, axis=1)
    mean4, logvar4 = tf.split(self.encoder4(x), num_or_size_splits=2, axis=1)
    mean5, logvar5 = tf.split(self.encoder5(x), num_or_size_splits=2, axis=1)
    # mean6, logvar6 = tf.split(self.encoder6(x), num_or_size_splits=2, axis=1)
    # mean7, logvar7 = tf.split(self.encoder7(x), num_or_size_splits=2, axis=1)
    # mean8, logvar8 = tf.split(self.encoder8(x), num_or_size_splits=2, axis=1)
    # mean9, logvar9 = tf.split(self.encoder9(x), num_or_size_splits=2, axis=1)
    # mean10, logvar10 = tf.split(self.encoder10(x), num_or_size_splits=2, axis=1)
    coeffs = self.coeff(x)
    return mean1, logvar1, mean2, logvar2, mean3, logvar3, mean4, logvar4, mean5, logvar5, coeffs


  def reparameterize(self, mean1, logvar1, mean2, logvar2, mean3, logvar3, mean4, logvar4, mean5, logvar5, coeffs):
    eps = tf.random.normal(shape=mean1.shape)
    eps1 = (eps * tf.exp(logvar1 * .5) + mean1)*coeffs[:, 0]
    eps2 = (eps * tf.exp(logvar2 * .5) + mean2)*coeffs[:, 1]
    eps3 = (eps * tf.exp(logvar3 * .5) + mean3)*coeffs[:, 2]
    eps4 = (eps * tf.exp(logvar4 * .5) + mean4)*coeffs[:, 3]
    eps5 = (eps * tf.exp(logvar5 * .5) + mean5)*coeffs[:, 4]
    return eps1 + eps2 + eps3 + eps4 + eps5

  def decode(self, z):
    mean, logvar = tf.split(self.decoder(z), num_or_size_splits=2, axis = 1)
    return mean, logvar

optimizer = tf.keras.optimizers.Adam(1e-6)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis, keepdims = True)


def compute_loss(model, x):
  mean1, logvar1, mean2, logvar2, mean3, logvar3, mean4, logvar4, mean5, logvar5, coeffs  = model.encode(x)
  z = model.reparameterize(mean1, logvar1, mean2, logvar2, mean3, logvar3, mean4, logvar4, mean5, logvar5, coeffs)
  xmean, xlogvar = model.decode(z)
  #conditional probability
  logpx_z = log_normal_pdf(x, xmean, xlogvar)
  #prior
  logpz = log_normal_pdf(z, 0., 0.)
  #posterior probability
  logq1 = log_normal_pdf(z, mean1, logvar1)
  logq2 = log_normal_pdf(z, mean2, logvar2)
  logq3 = log_normal_pdf(z, mean3, logvar3)
  logq4 = log_normal_pdf(z, mean4, logvar4)
  logq5 = log_normal_pdf(z, mean5, logvar5)
  logq = tf.concat([tf.math.log(coeffs[:,0])+logq1, tf.math.log(coeffs[:,1])+logq2, tf.math.log(coeffs[:,2])+logq3, tf.math.log(coeffs[:,3])+logq4, tf.math.log(coeffs[:,4])+logq5], axis = 1)
  logqz_x = tf.reduce_max(logq, axis = 1)
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
mean1, logvar1 = tf.split(model.encoder1(test), num_or_size_splits=2, axis=1)
mean2, logvar2 = tf.split(model.encoder2(test), num_or_size_splits=2, axis=1)
mean3, logvar3 = tf.split(model.encoder3(test), num_or_size_splits=2, axis=1)
mean4, logvar4 = tf.split(model.encoder4(test), num_or_size_splits=2, axis=1)
mean5, logvar5 = tf.split(model.encoder5(test), num_or_size_splits=2, axis=1)
all_coeff = model.coeff(test)
means = all_coeff[:,0:1]*mean1 + all_coeff[:,1:2]*mean2 + all_coeff[:,2:3]*mean3 + all_coeff[:,3:4]*mean4 + all_coeff[:,4:5]*mean5

#use fastICA to extract mixing matrix
np.savetxt("SimData/mean4.csv", means, delimiter=",")
#use fastICA to extract mixing matrix

from sklearn.decomposition import FastICA     
va_transformer = FastICA(n_components = 3, random_state = 0)
va_features = va_transformer.fit_transform(means.numpy().T).T

raw_transformer = FastICA(n_components = 3, random_state = 0)
raw_features = raw_transformer.fit_transform(test.T)

#save the estimated mixing matrix
#save the estimated mixing matrix
va_mixing = np.array(va_transformer.components_)
np.savetxt("SimData/sim4va.csv", va_features, delimiter=",")

raw_mixing = np.array(raw_transformer.components_)
np.savetxt("SimData/sim4raw.csv", raw_features, delimiter=",")


va_mixing = np.array(va_transformer.components_)
np.savetxt("SimData/sim4vamix.csv", va_mixing, delimiter=",")

raw_mixing = np.array(raw_transformer.components_)
np.savetxt("SimData/sim4rawmix.csv", raw_mixing, delimiter=",")