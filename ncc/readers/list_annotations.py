import os
from glob import glob


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

    annotation_set = [[image_file, label] for image_file, label in zip(image_files, labels)]

    return annotation_set


def classification_set(target_dir, train_dirs, test_dirs):
    """
    collect annotation files in target dir (target_dir/data_dir/class_dir/image_file)
    :param target_dir: root path that contains data set
    :param train_dirs: directory list used for train data
    :param test_dirs: directory list used for test data
    :return: train_set: [image_file_path, label_idx]
             test_set: [image_file_path, label_idx]
    """
    train_set, test_set = list(), list()

    data_dirs = os.listdir(target_dir)
    for data_dir in data_dirs:
        if data_dir in train_dirs:
            train_set += list_files(os.path.join(target_dir, data_dir))
        elif data_dir in test_dirs:
            test_set += list_files(os.path.join(target_dir, data_dir))

    return train_set, test_set
