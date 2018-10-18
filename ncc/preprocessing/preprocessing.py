import numpy as np


def preprocess_input(x_array, y_array=None, one_hot=True):

    if len(x_array.shape) == 3:  # (num_samples, height, width)
        x_array = np.expand_dims(x_array, axis=3)  # (num_samples, height, width, 1)

    x_array = x_array.astype('float32')
    x_array /= 255

    if y_array is None:
        return x_array

    if one_hot and len(y_array) == 1:  # (num_samples, )
        class_index = np.unique(y_array)
        num_classes = len(class_index)
        y_array = np.eye(num_classes)[y_array]  # one hot: (num_samples, num_classes)

    y_array = y_array.astype('float32')

    return x_array, y_array
