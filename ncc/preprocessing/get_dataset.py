import numpy as np
import glob

from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img


def get_dataset(target_dir):
    # get dataset from target_dir
    file_list = glob.glob(target_dir + '/*/')
    # image load
    x_array, y_array = [], []
    for num in range(len(file_list)):
        for picture in list_pictures(file_list[num]):
            img = img_to_array(load_img(picture))
            x_array.append(img) # input image
            y_array.append(num) # label
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    return x_array, y_array
