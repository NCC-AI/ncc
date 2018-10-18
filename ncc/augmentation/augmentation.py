# coding: utf-8

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def augmentation_from_file(image_path):

    SATURATION = np.random.rand()  # 0.0 ~ 1.0
    CONTRAST = np.random.randint(500, 1000)/1000  # 0.5 ~ 1.0
    BRIGHTNESS = np.random.randint(500, 1000)/1000  # 0.5 ~ 1.0
    SHARPNESS = np.random.randint(0, 2000)/1000  # 0.0 ~ 2.0
    FLIP = np.random.choice([True, False])  # True or False
    MIRROR = np.random.choice([True, False])  # True or False
    BLUR = np.random.randint(0, 1000)/1000  # 0.0 ~ 1.0

    img = Image.open(image_path)

    # 彩度を変える
    saturation_converter = ImageEnhance.Color(img)
    img = saturation_converter.enhance(SATURATION)

    # コントラストを変える
    contrast_converter = ImageEnhance.Contrast(img)
    img = contrast_converter.enhance(CONTRAST)

    # 明度を変える
    brightness_converter = ImageEnhance.Brightness(img)
    img = brightness_converter.enhance(BRIGHTNESS)

    # シャープネスを変える
    sharpness_converter = ImageEnhance.Sharpness(img)
    img = sharpness_converter.enhance(SHARPNESS)

    if FLIP:
       img = ImageOps.flip(img)  # 上下反転
    if MIRROR:
       img = ImageOps.mirror(img)    # 左右反転

    img = img.filter(ImageFilter.GaussianBlur(BLUR))  # ガウシアンブラー

    return img
