from glob import glob
import cv2
import numpy as np
import pandas as pd
import re


def search_from_dir(target_dir):
    files = []
    for image_suffix in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
        image_paths = target_dir + '/*/*/*.' + image_suffix  # target_dir/train(test)/class_name/*.jpg
        files += glob(image_paths)

    return _median_size(files)


def search_from_annotation(annotation_file):
    annotation_df = pd.read_csv(annotation_file).values.T
    files = annotation_df[0]
    class_index = annotation_df[1]

    class_names = np.unique(class_index)

    return _median_size(files), class_names


def _median_size(files):
    if len(files) > 10000:
        files = files[:10000]  # ignore large image files

    height_list, width_list = [], []
    for file in files:
        if re.search('.jpg|jpeg|bmp|png', file):
            image = cv2.imread(file)
            height, width = image.shape[:2]
            height_list.append(height)
            width_list.append(width)
    height_median = np.median(height_list).astype('int')
    width_median = np.median(width_list).astype('int')

    return height_median, width_median
