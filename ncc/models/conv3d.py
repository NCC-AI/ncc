from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Input
from keras.models import Model

import numpy as np

from .util import inst_layers


# Convolution 3D layer
def Conv(filters=16, kernel_size=(3, 3, 3), activation='relu', input_shape=None):
    if input_shape:
        return Conv3D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation=activation,
                      input_shape=input_shape)
    else:
        return Conv3D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation=activation)


# Define Model
def conv3d(input_dim, num_classes):

    small_size = min(input_dim[:3])
    nb_convolution = 0

    while small_size > 1:
        small_size = small_size // 2
        nb_convolution += 1

    latent_dim = np.prod(input_dim[:3]) * (8*2**(nb_convolution-1)) // (8**nb_convolution) // 4

    layers = [
        Conv(8, input_shape=input_dim),
        MaxPooling3D()
    ]

    layers += [
        [
            Conv(8 * 2**layer_id),
            BatchNormalization(),
            MaxPooling3D(),
        ]
        for layer_id in range(1, nb_convolution)
    ]

    layers += [
        Flatten(),
        Dropout(0.25),
        Dense(latent_dim, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='prediction')
    ]

    x_in = Input(shape=input_dim, name='input')
    prediction = inst_layers(layers, x_in)
    model = Model(x_in, prediction)

    return model
