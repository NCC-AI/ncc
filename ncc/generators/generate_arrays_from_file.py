import numpy as np
import re

from keras.preprocessing.image import load_img, img_to_array


def generate_arrays_from_annotation(annotation_data, dimension, batch_size, nb_classes, target_size):
    while True:

        x, y = [], []
        i = 0

        np.random.shuffle(annotation_data)
        for index, label_feature in enumerate(annotation_data):
            feature = _set_input_data(label_feature[0], target_size, dimension)
            y.append(label_feature[1])
            x.append(feature)
            i += 1
            if i == batch_size:
                yield (np.array(x), np.eye(nb_classes)[np.array(y)])
                i = 0
                x, y = [], []


def _set_input_data(image_path, target_size, dimension, nb_frames=32):
    # color order: (R, G, B)
    if dimension == 2:
        image = load_img(image_path, target_size)
        array = img_to_array(image)

    # stack sequence image with index (e.g. from image-4.jpg to image-36.jpg)
    elif dimension == 3:
        array = []
        first_frame_index = int(re.findall('(\d+)', image_path)[-1])

        for frame_index in range(first_frame_index, first_frame_index + nb_frames):
            sequence_path = image_path.replace(str(first_frame_index), str(frame_index))
            frame = load_img(sequence_path, target_size)
            array.append(img_to_array(frame))

        array = np.asarray(array)

    else:
        raise ValueError('invalid dimension arg. dimension should be 2 or 3')

    return array
