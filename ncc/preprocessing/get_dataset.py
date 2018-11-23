import numpy as np
import glob

from PIL import Image
from keras_preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from ncc.readers.search_image_size import search_from_dir

def get_dataset(target_dir, interpolation): #target_dir/train(or test)/class_name/*.jpg
    # get dataset from target_dir
    file_list = glob.glob(target_dir + '/*/')
    # image load
    x_array, y_array = [], []
    height_median, width_median = search_from_dir(target_dir) #get median size

    for class_index, folder_name in enumerate(file_list):
        for picture in list_pictures(folder_name):
            img = load_img(picture,target_size=(width_median ,height_median),interpolation=interpolation) #img type = PIL.image
            img_array = img_to_array(img) #np.array
            x_array.append(img_array) # input image
            y_array.append(class_index) # label

    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    return x_array, y_array
