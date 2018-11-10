import os
import csv
import cv2
import numpy as np

from keras.datasets import cifar10


def prepare_data(nb_image=100):

    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    _find_and_save_image(x_train, y_train, nb_image, names, 'data_set/train')
    _find_and_save_image(x_test, y_test, nb_image, names, 'data_set/test')


def _find_and_save_image(images, labels, nb_image, names, phase_folder):

    annotation = [['file_path', 'class_index']]
    for class_index, class_name in enumerate(names):
        print('find and save ', class_name)

        # save image in '.png' file
        save_dir = os.path.join(phase_folder, class_name)
        os.makedirs(save_dir, exist_ok=True)

        cls_image = images[np.where(labels.ravel() == class_index)]
        np.random.shuffle(cls_image)

        for i, image in enumerate(cls_image[:nb_image]):
            file_path = os.path.join(save_dir, str(i) + '.png')
            cv2.imwrite(file_path, image)
            annotation.append([file_path, class_index])

    with open(phase_folder+'_annotation.csv', 'w') as fw:
        print(phase_folder, 'annotation file saved in data_set/annotation.csv')
        writer = csv.writer(fw)
        writer.writerows(annotation)
