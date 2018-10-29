from glob import glob
import cv2
import numpy as np

from ncc.preprocessing import segmentation_input, preprocess_input
from ncc.models import Unet
from ncc.history import save_history

# parameters
epochs = 30
batch_size = 4

# prepare data

root_data_path = '/home/yoshiki_watanabe/Desktop/project/BloodSegmentation/data'
label_data_path = root_data_path + '/label/*.png'

label_files = glob(label_data_path)

in_image = []
l_image = []
for label_file in label_files:
    label_image = cv2.imread(label_file)
    raw_image_path = label_file.replace('label', 'movieFrame')
    raw_image = cv2.imread(raw_image_path)

    in_image.append(raw_image)
    l_image.append(segmentation_input(label_image))

l_image = np.asarray(l_image, dtype='float32')

# build model
model, input_shape = Unet(input_shape=in_image[0].shape, output_channel_count=l_image.shape[-1])
model.summary()

# resize
if input_shape != in_image[0].shape:
    in_image = [cv2.resize(image, input_shape[:2]) for image in in_image]

in_image = preprocess_input(np.asarray(in_image))
print(in_image.shape, l_image.shape)

# train
history = model.fit(in_image, l_image, epochs=epochs, batch_size=batch_size)
save_history(history)
