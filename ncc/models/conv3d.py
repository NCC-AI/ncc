from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, BatchNormalization


# Conv3D layer
def Conv(filters=16, kernel_size=(3, 3, 3), activation='relu', input_shape=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)


# Define Model
def conv3d(input_dim, num_classes):
    model = Sequential()

    model.add(Conv(8, (3, 3, 3), input_shape=input_dim))
    model.add(MaxPool3D())
    model.add(Conv(16, (3, 3, 3)))
    model.add(MaxPool3D())
    model.add(Conv(32, (3, 3, 3)))
    model.add(MaxPool3D())
    model.add(Conv(64, (3, 3, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model
