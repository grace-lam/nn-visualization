## Import packages for building neural networks 

import numpy as np
import os
from math import floor, ceil

import tensorflow as tf
import keras.backend as K

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Reshape
from keras.layers import Lambda
from keras.layers import Conv2DTranspose
from keras.models import Model

## Build a neural networks model (for supervised training)

latent_dim = 128*128

### Residual blocks

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

### Encoder

def encoder_variational(input_shape, num_classes=3):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2 (except that the stride for the first stage is 4), 
    while the number of filters is doubled. 
    Within each stage, the layers have the same number filters 
    and the same number of filters.
    Features maps sizes:
    stage 0: 256x256, 8
    stage 1: 128x128,  16
    stage 2: 64x64,  32
    stage 3: 32x43,  64
    stage 4: 16x16,   128

    output:   128x128,   2
    
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    # if (depth - 2) % 10 != 0:
    #     raise ValueError('depth should be 10n+2 (eg 22, 32, 42 in [a])')
    # Start model definition.
    num_filters = 8
    num_res_blocks = 2 #int((depth - 2) / 10)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, kernel_size=7, 
                     num_filters=num_filters, strides=4)
    # Instantiate the stack of residual units
    for stack in range(5):
        if stack == 0:
            kernel_size = 5
        else:
            kernel_size = 3
        for res_block in range(num_res_blocks):
            strides = 1
            if res_block == 0:  # first layer of each stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             kernel_size = kernel_size,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             kernel_size = kernel_size,
                             activation=None)
            if res_block == 0:  # first layer
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = resnet_layer(inputs=x, kernel_size=1,
                     num_filters=128, strides=1,
                     activation=None, batch_normalization=False)

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=2)(x)
    # y = Flatten()(x)
    # y = Dense(64,
    #           kernel_initializer='he_normal')(y)
    # y = BatchNormalization()(y)
    # y = Activation('relu')(y)
    # outputs = Dense(num_classes,
    #                 activation='sigmoid',
    #                 kernel_initializer='he_normal')(y)

    z_mean_log_var = Flatten()(x)
    #z_mean_log_var = Dense(latent_dim*2, name='z_mean_log_var')(x)
    #z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    #latent_z = Lambda(sampling, name='latent_z')([z_mean, z_log_var])

    # Instantiate model.
    model = Model(inputs=inputs, outputs=z_mean_log_var, 
                  name='encoder_variational')
    
    return model


### Decoder

def decoder(input_shape=(latent_dim, )):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2 (except that the stride for the first stage is 4), 
    while the number of filters is doubled. 
    Within each stage, the layers have the same number filters 
    and the same number of filters.
    Features maps sizes:
    stage 0: 16x16    128
    stage 1: 32x32     64
    stage 2: 64x64     32
    stage 3: 128x128   16
    stage 4: 256x256    8
    stage 5: 512x512    4
    stage 6: 1024x1024  2
    stage 7: 2048x2048  1

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    # if (depth - 2) % 10 != 0:
    #     raise ValueError('depth should be 10n+2 (eg 22, 32, 42 in [a])')
    # Start model definition.
    num_filters = 256
    kernel_size = 3

    inputs = Input(shape=input_shape)
    x = Reshape((8, 8, num_filters))(inputs)

    # Instantiate the stack of residual units
    for stack in range(8):
        num_filters //= 2
        strides = 2
        x = Conv2DTranspose(filters=num_filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=strides,
                            padding='same')(x)

    outputs = x

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name='decoder')
    
    return model


### Classifier

def classifier(input_shape=(latent_dim, ), num_classes=3):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2 (except that the stride for the first stage is 4), 
    while the number of filters is doubled. 
    Within each stage, the layers have the same number filters 
    and the same number of filters.
    Features maps sizes:
    stage 0: 64x64     8
    stage 1: 32x32    16
    stage 2: 16x16    32
    stage 3: 8x8      64
    stage 4: 4x4      128
    
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    inputs = Input(shape=input_shape)
    x = Reshape((128, 128, 1))(inputs)

    num_filters = 8
    num_res_blocks = 2
    kernel_size = 3

    # Instantiate the stack of residual units
    for stack in range(5):
        for res_block in range(num_res_blocks):
            strides = 1
            if res_block == 0:  # first layer of each stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             kernel_size = kernel_size,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             kernel_size = kernel_size,
                             activation=None)
            if res_block == 0:  # first layer
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2


    x = AveragePooling2D(pool_size=2)(x)
    y = Flatten()(x)
    y = Dense(64,
              kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)
    
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name='classifier')
    
    return model