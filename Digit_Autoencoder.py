import numpy as np
import matplotlib.pyplot as plt
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

## Custom padding that pads with zeros on right and bottom
## Used in Decoder to grow size from (6,6,32) to (7,7,32)
class Custom_Padding(layers.Layer):

    def __init__(self, name='custom_padding'):
        super(Custom_Padding,self).__init__(name=name)

    def call(self, inputs):
        shape = inputs.shape
        zeros_1 = tf.zeros((shape[0],1,shape[2],shape[3]))
        one_axis = tf.concat([inputs,zeros_1],1)
        shape = one_axis.shape
        zeros_2 = tf.zeros((shape[0],shape[1],1,shape[3]))
        return tf.concat([one_axis,zeros_2],2)


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
        self.pool_3 = layers.MaxPooling2D((2,2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation='relu')

    def call(self, inputs):
        conv_1 = self.conv_1(inputs)
        pool_1 = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool_1)
        pool_2 = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool_2)
        pool_3 = self.pool_3(conv_3)
        flatten = self.flatten(pool_3)
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
        
        self.dense  = layers.Dense(dim_3*3*3, activation='relu')
        self.reshape = layers.Reshape((3,3,dim_3))
        self.upsample_2 = layers.UpSampling2D((2,2))
        self.padding = Custom_Padding()
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
        padding = self.padding(upsample_2)
        convT_2 = self.convT_2(padding)
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




(train, _ ) , (test , _ ) = datasets.mnist.load_data()

## How many images to display
n = 10
plt.figure(figsize=(10 , 2))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

ae = Autoencoder(16,32,64)
ae.build((1,28,28,1))
ae.summary()

ae.compile(optimiser='adam', loss=ae.loss)


