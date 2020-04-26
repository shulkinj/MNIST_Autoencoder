import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets, losses 
import datetime

tf.keras.backend.clear_session()  # For easy reset of notebook state.
"""
# Load the TensorBoard notebook extension
%load_ext tensorboard

# Clear any logs from previous runs
!rm -rf ./logs/
"""

class Encoder(layers.Layer):

    def __init__(self,
                dim_1=16,
                dim_2=32,
                dim_3=64,
                name='encoder',
                **kwargs):
        super(Encoder,self).__init__(name=name, **kwargs)
        self.conv_1 = layers.Conv2D(dim_1, (3,3) , strides=(1,1), padding='same', activation='relu',
                      input_shape=(28, 28, 1))
        self.pool_1 = layers.MaxPooling2D((2,2))
        self.conv_2 = layers.Conv2D(dim_2, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.pool_2 = layers.MaxPooling2D((2,2))
        self.conv_3 = layers.Conv2D(dim_3, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation='relu')

    def call(self, inputs):
        conv_1 = self.conv_1(inputs)
        pool_1 = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool_1)
        pool_2 = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool_2)
        flatten = self.flatten(conv_3)
        dense = self.dense(flatten)
        return dense



class Decoder(layers.Layer):

    def __init__(self,
                dim_3=64,
                dim_2=32,
                dim_1=16,
                name='decoder',
                **kwargs):
        super(Decoder,self).__init__(name=name, **kwargs)
        
        self.dense  = layers.Dense(dim_3*7*7, activation='relu')
        self.reshape = layers.Reshape((7,7,dim_3))
        self.convT_3 = layers.Conv2DTranspose(dim_3, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.upsample_2 = layers.UpSampling2D((2,2))
        self.convT_2 = layers.Conv2DTranspose(dim_2, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.upsample_1 = layers.UpSampling2D((2,2))
        self.convT_1 = layers.Conv2DTranspose(dim_1, (3,3) , strides=(1,1), padding='same', activation='relu')


    def call(self, inputs):
        dense = self.dense(inputs)
        reshape = self.reshape(dense)
        convT_3 = self.convT_3(reshape)
        upsample_2 = self.upsample_2(convT_3)
        convT_2 = self.convT_2(upsample_2)
        upsample_1 = self.upsample_1(convT_2)
        convT_1 = self.convT_1(upsample_1)
        return convT_1


class Autoencoder(tf.keras.Model):

    def __init__(self,
                dim_1 = 16,
                dim_2 = 32,
                dim_3 = 64,
                name = 'autoencoder',
                **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(dim_1=dim_1, dim_2=dim_2, dim_3=dim_3)
        self.decoder = Decoder(dim_3=dim_3, dim_2=dim_2, dim_1=dim_1)


    def call(self, inputs):
        representation = self.encoder(inputs)
        reconstruction = self.decoder(representation)
        self.add_loss(lambda : losses.MSE(inputs, reconstruction))
        return reconstruction


ae = Autoencoder(16,32,64)
ae.build((1,28,28,1))
ae.summary()

