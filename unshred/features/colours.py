from .base import AbstractShredFeature
import json
import cv2
from array import array
import os.path
from colourdistance import closest_by_palette
from collections import Counter


def hex_to_bgr(hex_digits):
    """
    Convert a hexadecimal color value to a 3-tuple of integers
    """
    return tuple(int(s, 16) for s in (
        hex_digits[:2], hex_digits[2:4], hex_digits[4:]))


class ColourFeatures(AbstractShredFeature):
    def __init__(self, *args, **kwargs):
        super(ColourFeatures, self).__init__(*args, **kwargs)
        fname = os.path.join(os.path.dirname(__file__), "palette.json")

        with open(fname, "r") as fp:
            self.palette = json.load(fp)

        self.reverse_palette = {}
        self.flat_palette = []
        self.bands = []
        for band, colours in self.palette.iteritems():
            self.bands.append(band)
            for colour in map(hex_to_bgr, colours):
                self.reverse_palette[colour] = len(self.bands) - 1
                self.flat_palette.append(colour)

        self.generate_mapping()

    def generate_mapping(self):
        cache_fname = os.path.join(os.path.dirname(__file__), "mapping.pickle")
        if os.path.exists(cache_fname):
            with open(cache_fname, "rb") as fp:
                self.plt = array("B")
                self.plt.fromfile(fp, 1 << 24)
            return

        self.plt = array("B")
        for b in xrange(0, 256):
            print(b)
            for g in xrange(0, 256):
                for r in xrange(0, 256):
                    self.plt.append(self.reverse_palette[
                        self.flat_palette[
                            closest_by_palette((r, g, b), self.flat_palette)]])

        with open(cache_fname, "wb") as fp:
            self.plt.tofile(fp)

    def convert_colour(self, px):
        if px[3] == 0:
            return 'transparent'
        else:
            return self.bands[self.plt[(px[0] << 16) + (px[1] << 8) + px[2]]]

    def get_info(self, shred, contour):
        w, h, _ = shred.shape
        w = int(w / 3)
        h = int(h / 3)

        resized = cv2.resize(shred, (w, h),
                             interpolation=cv2.INTER_AREA)

        cnt = Counter(self.convert_colour(resized[x, y])
                      for x in xrange(h)
                      for y in xrange(w))

        print(cnt)
        return {}, ()
