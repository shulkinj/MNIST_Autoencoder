import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, datasets, losses , models
import datetime

tf.keras.backend.clear_session()  # For easy reset of notebook state.





## Custom padding that pads with zeros on right and bottom
## Used in Decoder to grow size from (6,6,32) to (7,7,32)
class Custom_Padding(layers.Layer):

    def __init__(self, name='custom_padding'):
        super(Custom_Padding,self).__init__(name=name)

    def call(self, inputs):
        dims = tf.stack([tf.shape(inputs)[0], 1, 
                        tf.shape(inputs)[2], tf.shape(inputs)[3]])
        zeros_1 = tf.fill(dims, 0.0)
        one_axis = tf.concat( values=[inputs,zeros_1] , axis=1)
        dims = tf.stack([tf.shape(one_axis)[0], tf.shape(one_axis)[1], 
                        1, tf.shape(one_axis)[3]])
        zeros_2 = tf.fill(dims, 0.0)
        padded = tf.concat(values= [one_axis,zeros_2], axis= 2)
        return padded


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
        self.upsample_3 = layers.UpSampling2D((2,2))
        self.padding = Custom_Padding()
        self.convT_3 = layers.Conv2DTranspose(dim_2, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.upsample_2 = layers.UpSampling2D((2,2))
        self.convT_2 = layers.Conv2DTranspose(dim_1, (3,3) , strides=(1,1), padding='same', activation='relu')
        self.upsample_1 = layers.UpSampling2D((2,2))
        self.convT_1 = layers.Conv2DTranspose(1, (3,3) , strides=(1,1), padding='same', activation='relu')


    def call(self, inputs):
        dense = self.dense(inputs)
        reshape = self.reshape(dense)
        upsample_3 = self.upsample_3(reshape)
        padding = self.padding(upsample_3)
        convT_3 = self.convT_3(padding)
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
        return reconstruction



###### Load Data ######
(train, _ ) , (test , _ ) = datasets.mnist.load_data()

train = train.astype('float32')/255.0
test = test.astype('float32')/255.0

train_shape = (train.shape[0],train.shape[1],train.shape[2],1)
test_shape = (test.shape[0],test.shape[1],test.shape[2],1)

train = np.reshape(train,train_shape)
test = np.reshape(test,test_shape)


###### Add noise ######
noise_factor = 0.3
train_noisy = train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train.shape) 
test_noisy = test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test.shape) 

train_noisy = np.clip(train_noisy, 0., 1.)
test_noisy = np.clip(test_noisy, 0., 1.)



###### Set up Model ######
batch_sz=32

ae = Autoencoder(16,32,64)


opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
ae.compile(optimizer=opt, loss=losses.MSE)


## Tensorboard set up ##
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


###### TRAINING ######
ae.fit( train_noisy, train, epochs=20, batch_size=batch_sz, 
        shuffle= True, validation_data=(test_noisy,test),
        callbacks=[tensorboard_callback])

ae.save('Saved_Model/denoiser')



#ae = models.load_model('Saved_Model/denoiser')


###### Display Test inputs and outputs #######
decoded_imgs = ae.predict(test_noisy)

n=20
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

