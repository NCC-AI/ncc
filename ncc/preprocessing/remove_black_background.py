""" Remove black line in images and save processed images.
"""
import numpy as np


def process(img):
    h_start, h_end = get_start_end(img, 'horizontal')
    tmp_img = rm_black(img, h_start, h_end, 'horizontal')

    v_start, v_end = get_start_end(tmp_img, 'vertical')
    clipped_img = rm_black(tmp_img, v_start, v_end, 'vertical')

    return clipped_img


def get_start_end(img, mode):
    """
    # Returns
        start: int, None if not expected
        end: int, None if not expected
    """
    shape = img.shape
    if mode == 'horizontal':
        checked_seq = img[shape[0]//2][:]

    elif mode == 'vertical':
        checked_seq = img[:][shape[1]//2]

    else:
        raise ValueError('model is horizontal or vertical')

    start = None
    end = None
    bound = {'horizontal': 20, 'vertical': 40}
    for idx, pixel in enumerate(checked_seq):
        if not np.all(pixel < bound[mode]):
            start = idx
            break
    for idx, pixel in enumerate(checked_seq[::-1]):
        if not np.all(pixel < bound[mode]):
            if mode == 'horizontal':
                end = shape[1] - idx
            elif mode == 'vertical':
                end = shape[0] - idx
            break

    return start, end


def rm_black(img, start, end, mode):
    """
    # Args
        mode: string, 'horizontal' or 'vertical'
    """
    if mode == 'horizontal':
        clipped_img = img[:, start:end]

    elif mode == 'vertical':
        clipped_img = img[start:end, :]

    else:
        raise ValueError('model is horizontal or vertical')

    return clipped_img
