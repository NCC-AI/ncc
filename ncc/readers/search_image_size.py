from glob import glob
import cv2
import numpy as np


def search_image_size(target_dir):
    files = []
    for image_suffix in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
        image_paths = target_dir + '/*/*/*.' + image_suffix  # target_dir/trained/class_name/*.jpg
        files += glob(image_paths)

    return median_size(files)


def median_size(files):
    if len(files) > 10000:
        files = files[:10000]  # ignore large image files

    height_list, width_list = [], []
    for file in files:
        image = cv2.imread(file)
        height, width = image.shape[:2]
        height_list.append(height)
        width_list.append(width)
    height_median = np.median(height_list)
    width_median = np.median(width_list)

    return height_median, width_median
