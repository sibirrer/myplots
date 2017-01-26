__author__ = 'sibirrer'

import aplpy
from PIL import Image


class ColorPlot(object):
    """
    class for multiband color plot
    """

    def make_rgb_image(self, image_r, image_g, image_b):
        image_r = aplpy._data_stretch(image_r)
        image_g = aplpy._data_stretch(image_g)
        image_b = aplpy._data_stretch(image_b)
        image_r = Image.fromarray(image_r, mode="RGB")
        image_g = Image.fromarray(image_g, mode="RGB")
        image_b = Image.fromarray(image_b, mode="RGB")
        img = Image.merge("RGB", (image_r, image_g, image_b))
        return img

