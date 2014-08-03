import cv2
import itertools
from PIL import Image

from .base import AbstractShredFeature


def hex_to_bgr(hex_digits):
    """
    Convert a hexadecimal color value to a 3-tuple of integers
    """
    return tuple(int(s, 16) for s in (
        hex_digits[:2], hex_digits[2:4], hex_digits[4:]))


class ColourFeatures(AbstractShredFeature):
    palette_colours = [
        "000000",
        "434343",
        "666666",
        "999999",
        "b7b7b7",
        "cccccc",
        "d9d9d9",
        "efefef",
        "f3f3f3",
        "ffffff",
        "96000a",
        "fc0019",
        "fc992b",
        "fcff42",
        "06ff3b",
        "2cfffe",
        "5284e4",
        "2a00f9",
        "a4170e",
        "ca0012",
        "efc347",
        "69a956",
        "48818d",
        "4676d4",
        "4484c3",
        "694ba4",
        "a54b78",
        "831e13",
        "97000a",
        "b15f1b",
        "bd9024",
        "387727",
        "184f5b",
        "2652c8",
        "1a5290",
        "371872",
        "731746",
        "5a0d04",
        "650004",
        "773f10",
        "7e6015",
        "274f19",
        "0f343b",
        "234484",
        "103661",
        "22104b",
        "4b0f2f",
    ]

    skip = 10  # Amount of colours from the beginning of the list to ignore
               # white, black and shades of gray.

    def __init__(self, *args, **kwargs):
        super(ColourFeatures, self).__init__(*args, **kwargs)

        # Building our special palette.
        # Pillow is strange when it goes about palettes.
        # Here we are creating a new image and add some colours to it's
        # palette and rest of unused 256 is set to white. Otherwise
        # PIL will use shades of gray to fill unused ones.
        self.palette = Image.new("P", (1, 1), 0)

        # convert colours to the form suitable for set_palette
        plt = list(itertools.chain.from_iterable(
                   map(hex_to_bgr, self.palette_colours)))

        # padding the rest of palette with white colour
        self.palette.putpalette(
            plt + [255, 255, 255] * (256 - len(self.palette_colours)))

    def has_blue_ink(self, histogram):
        h = dict(zip(self.palette_colours, histogram))
        ratio = (h["694ba4"] + h["5284e4"] +
                 h["2a00f9"]) / float(sum(histogram))

        # Too much of blue inc is also bad (usually means a picture)
        if (ratio >= 0.01 and ratio <= 0.1 and
                not self.part_of_a_picture(histogram)):
            return True

    def part_of_a_picture(self, histogram):
        if sum(histogram[self.skip:]) / float(sum(histogram)) > 0.2:
            return True

    def has_yellow_marks(self, histogram):
        h = dict(zip(self.palette_colours, histogram))
        ratio = (h["efc347"] + h["fcff42"]) / float(sum(histogram))

        # Too much of blue inc is also bad (usually means a picture)
        if (ratio >= 0.05):
            return True

    # List of heuristics for particular features.
    tag_detectors = {
        "Has blue ink": has_blue_ink,
        "Part of a picture": part_of_a_picture,
        "Has yellow marks": has_yellow_marks,
    }

    def get_info(self, shred, contour, name):
        shred_pil = Image.fromarray(cv2.cvtColor(shred, cv2.COLOR_BGR2RGB))

        # downsampling image to our palette
        converted = shred_pil.quantize(palette=self.palette)
        # converted.save("../debug/converted%s.png" % name)

        mask = Image.fromarray(shred[:, :, 3])
        # Converting histogram with mask applied
        hist = converted.histogram(mask=mask)[:len(self.palette_colours)]

        total_pixels = sum(hist)
        hist_to_store = [0] * self.skip
        dominant_colours = []

        # Preparing clean version of histogram where only colours that has
        # > 1% is included. Same goes for the list of dominant colours.
        # Of course we are skipping white/black and shades of gray.
        for i, x in enumerate(self.palette_colours[self.skip:]):
            if hist[i + self.skip] > 0.01 * total_pixels:
                hist_to_store.append(hist[i + self.skip])
                dominant_colours.append(x)
            else:
                hist_to_store.append(0)

        tags = []
        for tag_name, tag_detector in self.tag_detectors.iteritems():
            if tag_detector(self, hist):
                tags.append(tag_name)

        return {
            "histogram_clean": hist_to_store,
            "histogram_full": hist,
            "colour_names": self.palette_colours,
            "dominant_colours": dominant_colours
        }, tuple(tags)
