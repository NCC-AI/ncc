# coding: utf-8
import os

from keras.models import Model
from keras.layers import Input, Cropping2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
import keras.backend as K

# self made function
from keras.utils import plot_model


def summary_and_png(model, summary=True, to_png=False, png_file=None):
    if summary:
        model.summary()
    if to_png:
        os.makedirs('summary', exist_ok=True)
        plot_model(model, to_file='summary/'+png_file, show_shapes=True)


class Unet(object):
    def __init__(self, input_shape, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_HEIGHT = input_shape[0]
        self.INPUT_IMAGE_WIDTH = input_shape[1]
        self.CHANNEL_COUNT = input_shape[2]
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.OUPUT_CHANNEL_COUNT = output_channel_count
        self.First_Filter_Count = first_layer_filter_count

    def model_720_1280(self):

        # (720 x 1280 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_HEIGHT, self.INPUT_IMAGE_WIDTH, self.CHANNEL_COUNT))

        # エンコーダーの作成
        # (360 x 640 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(self.First_Filter_Count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (180 x 320 x 2N)
        filter_count = self.First_Filter_Count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (90 x 160 x 4N)
        filter_count = self.First_Filter_Count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (45 x 80 x 8N)
        filter_count = self.First_Filter_Count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (22 x 40 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (11 x 20 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (5 x 10 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (2 x 5 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (4 x 10 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, enc8)
        ch, cw = self._get_crop_shape(enc7, dec1)
        crop_enc7 = Cropping2D(cropping=(ch, cw))(enc7)
        dec1 = concatenate([dec1, crop_enc7], axis=self.CONCATENATE_AXIS)

        # (11 x 20 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = Conv2DTranspose(filter_count, (4, 1), strides=1,
                               kernel_initializer='he_uniform')(dec2)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (22 x 40 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (45 x 80 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = Conv2DTranspose(filter_count, (2, 1), strides=1,
                               kernel_initializer='he_uniform')(dec4)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (90 x 160 x 4N)
        filter_count = self.First_Filter_Count*4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (180 x 320 x 2N)
        filter_count = self.First_Filter_Count*2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (360 x 640 x N)
        filter_count = self.First_Filter_Count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (720 x 1280 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(self.OUPUT_CHANNEL_COUNT, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        return Model(input=inputs, output=dec8)

    def model_256_256(self):
        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_HEIGHT, self.INPUT_IMAGE_WIDTH, self.CHANNEL_COUNT))

        # エンコーダーの作成
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(self.First_Filter_Count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        filter_count = self.First_Filter_Count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)
        filter_count = self.First_Filter_Count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)
        filter_count = self.First_Filter_Count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, enc8)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)
        filter_count = self.First_Filter_Count*4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)
        filter_count = self.First_Filter_Count*2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = self.First_Filter_Count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(self.OUPUT_CHANNEL_COUNT, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        return Model(input=inputs, output=dec8)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self, summary=True, to_png=False, png_file=None):
        if self.INPUT_IMAGE_HEIGHT == 720 and self.INPUT_IMAGE_WIDTH == 1280:
            unet_model = self.model_720_1280()
        elif self.INPUT_IMAGE_HEIGHT == 256 and self.INPUT_IMAGE_WIDTH == 256:
            unet_model = self.model_256_256()
        else:
            raise Excpetion('Invalid Input Shape')
        summary_and_png(unet_model, summary, to_png, png_file)
        return unet_model

    @staticmethod
    def _get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target._keras_shape[2] - refer._keras_shape[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target._keras_shape[1] - refer._keras_shape[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
