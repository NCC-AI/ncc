import colorsys
import numpy as np


class ImageUtil:

    def __init__(self):
        self.palette = self._make_palette()

    @staticmethod
    def random_colors(nb_colors, bright=True, scale=True, shuffle=False):
        """ Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / nb_colors, 1, brightness) for i in range(nb_colors)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        if scale:
            colors = tuple(np.array(np.array(colors)*255, dtype=np.uint8))
        if shuffle:
            colors = list(colors)
            np.random.shuffle(colors)
            colors = tuple(colors)
        return colors

    @staticmethod
    def apply_mask(image, mask, color, alpha=0.5):
        """ Apply the given mask to the image.
        image: (height, width, channel)
        mask: (height, width)
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def convert_to_palette(self, numpy_image):
        """Convert numpy gray scale image to pil palette"""
        pil_palette = Image.fromarray(np.uint8(numpy_image), mode="P")
        pil_palette.putpalette(self.palette)
        return pil_palette

    def _make_palette(self):
        colors = self.random_colors(256, shuffle=True)
        palette = []
        for color in colors:
            palette.extend(list(color))
        return palette
