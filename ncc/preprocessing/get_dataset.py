import numpy as np
from glob import glob
import os

from keras_preprocessing.image import img_to_array, load_img
from ncc.readers.search_image_size import search_from_dir


def get_dataset(target_dir, interpolation='nearest'):  # target_dir/class_name/*.jpg
    # get dataset from target_dir
    image_files, labels = list_files(target_dir)
    height_median, width_median = search_from_dir(target_dir)  # get median size
    images = list()
    for image_file in image_files:
        img = load_img(image_file, target_size=(width_median, height_median), interpolation=interpolation)
        img_array = img_to_array(img)
        images.append(img_array)

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels


def list_files(data_dir):
    """
    :param data_dir: this directory contains category directories
                    (data_dir/class_name/*.jpg)
    :return: file paths , label index
    """
    image_files = list()
    labels = list()
    class_names = os.listdir(data_dir)
    for class_id, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_ex in ['*.jpg', '*.png']:
            class_files = glob(os.path.join(class_dir, image_ex))
            image_files += class_files
            labels += [class_id for _ in range(len(class_files))]

    return image_files, labels
