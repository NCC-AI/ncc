from ncc.models import Model2D
from ncc.history import save_history
from ncc.preprocessing import preprocess_input

from keras.datasets import mnist

import numpy as np

num_classes = 10
depth = 8
width, height = 28, 28
channel = 1
input_shape = (width, height, channel)
epochs = 30
batch_size = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train, y_train = preprocess_input(x_train, y_train)
x_test, y_test = preprocess_input(x_test, y_test)


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Model2D(input_shape=input_shape, num_classes=num_classes)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

save_history(history)
