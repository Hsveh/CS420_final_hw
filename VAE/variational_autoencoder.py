'''
Variational autoencoder
SVM connected to the output of the encoder
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import common


def sampling(z_mean, z_log_var):
    """
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    :param z_mean: mean of Gaussian variable z
    :param z_log_var: covariance matrix of Gaussian variable z
    :return z sampled from z_mean and z_log_var
    """

    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# MNIST dataset
data = common.Data(common_path+"/mnist_train/train_data.npy", common_path+"/mnist_train/mnist_train_label",
                    common_path+"/mnist_test/test_data.npy", common_path+"/mnist_test/mnist_test_label", fig)
x_train = data.train_x
x_test = data.test_x
y_train = data.train_y
y_test = data.test_y

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 32
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')(z_mean, z_log_var)

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(64, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':

    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    # train the autoencoder
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    vae.save('vae_mnist.h5')

    # get extracted fetures
    train_output = encoder.predict(x_train)[0]
    test_output = encoder.predict(x_test)[0]

    print(train_output.shape, test_output.shape)
    print(type(train_output))

    # train SVM classifier
    clf=svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf.fit(train_output, y_train)
    pred_y = clf.predict(test_output)
    print(clf.score(test_output,y_test))

    prec = precision_score(y_test,pred_y,average='weighted')
    recall = recall_score(y_test,pred_y,average='weighted')
    f1 = f1_score(y_test,pred_y,average='weighted')

    print('precision:\t'+str(prec))
    print('recall:\t'+str(recall))
    print('f1 score:\t'+str(f1))
