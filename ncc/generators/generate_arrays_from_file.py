import numpy as np

from keras.preprocessing.image import load_img, img_to_array


def set_input_data(image_path, class_name, class_num, dimension, nb_frames=32):
    if dimension == 3:
        image = []
        first_image_path = image_path.split('/')
        image_dir = first_image_path[:-1]
        first_frame_index = int(first_image_path[-1].replace('.jpg', ''))
        for i in range(nb_frames):
            frame_index = first_frame_index + i
            frame_image_path = image_dir + [str(frame_index)]
            imaga_path = '/'.join(frame_image_path) + '.jpg'
            frame = load_img(image_path, target_size=(256, 256))
            array = img_to_array(frame)
            frame = array/255.
            image.append(frame)
        image = np.asanyarray(image)
        class_one_hot = np.eye(class_num)[int(class_name)-1]
    else:
        image = load_img(image_path, target_size=(299, 299))
        array = img_to_array(image)
        image = array/255.
        class_one_hot = np.eye(class_num)[int(class_name)-1]
    return image, class_one_hot


def generate_arrays_from_file(annotation_data, mode, dimension, batch_size, nb_classes):
    while True:
        x, y = [], []
        i = 0
        if mode == 'train':
            np.random.shuffle(annotation_data)
            for index, label_feature in enumerate(annotation_data):
                feature, label = set_input_data(label_feature[0], label_feature[1], nb_classes, dimension)
                y.append(label)
                x.append(feature)
                i += 1
                if i == batch_size:
                    yield (np.array(x), np.array(y))
                    i = 0
                    x, y = [], []
        if mode == 'test':
            np.random.shuffle(annotation_data)
            for index, label_feature in enumerate(annotation_data):
                feature, label = set_input_data(label_feature[0], label_feature[1], nb_classes, dimension)
                y.append(label)
                x.append(feature)
                i += 1
                if i == batch_size:
                    yield (np.array(x), np.array(y))
                    i = 0
                    x, y = [], []