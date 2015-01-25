from argparse import ArgumentParser
from glob import glob
import os
import os.path
import shutil

import cv2
import exifread
from jinja2 import FileSystemLoader, Environment
import numpy as np

from features import GeometryFeatures, ColourFeatures, LinesFeatures
from sheet import Sheet


parser = ArgumentParser()
parser.add_argument(
    'fnames', type=str, help='Input files glob.', nargs='?',
    default="../src/puzzle_small.tif")
parser.add_argument(
    'out_format', type=str, help='Output format.', nargs='?',
    default="png")


def convert_poly_to_string(poly):
    # Helper to convert openCV contour to a string with coordinates
    # that is recognizible by html area tag
    return ",".join(map(lambda x: ",".join(map(str, x[0])), poly))

env = Environment(loader=FileSystemLoader("templates"))
env.filters["convert_poly_to_string"] = convert_poly_to_string


class SheetIO(object):
    """Encapsulates IO operations on sheets and shreds."""

    def __init__(self, fname, sheet_name, feature_extractors,
                 out_dir="out", out_format="png"):

        self.fname = fname
        self.sheet_name = sheet_name
        self.out_format = out_format
        self.out_dir = out_dir

        orig_img = cv2.imread(fname)
        self.sheet = Sheet(orig_img, dpi=self._dpi_from_exif(),
                           save_image=self.save_image)
        self.feature_extractors = [
            feat(self.sheet) for feat in feature_extractors]

    def _dpi_from_exif(self):
        def parse_resolution(val):
            x = map(int, map(str.strip, str(val).split("/")))
            if len(x) == 1:
                return x[0]
            elif len(x) == 2:
                return int(round(float(x[0]) / x[1]))
            else:
                raise ValueError

        with open(self.fname, "rb") as f:
            tags = exifread.process_file(f)

        if "Image XResolution" in tags and "Image YResolution" in tags:
            try:
                return (parse_resolution(tags["Image XResolution"]),
                        parse_resolution(tags["Image YResolution"]))
            except ValueError:
                return None
        return None

    def save_image(self, fname, img, format=None):
        full_out_dir, fname = os.path.split(
            os.path.join(self.out_dir, self.sheet_name, fname))

        try:
            os.makedirs(full_out_dir)
        except OSError:
            pass

        if format is None:
            format = self.out_format

        fname = "%s/%s.%s" % (full_out_dir, fname, format)
        cv2.imwrite(fname, img)

        return fname

    def overlay_contours(self):
        # Draw contours on top of image with a nice yellow tint
        overlay = np.zeros(self.sheet.orig_img.shape, np.uint8)
        contours = map(lambda x: x.contour, self.get_shreds())

        # Filled yellow poly.
        cv2.fillPoly(overlay, contours, [104, 255, 255])
        img = self.sheet.orig_img.copy() + overlay

        # Add green contour.
        cv2.drawContours(img, contours, -1, [0, 180, 0], 2)
        return img

    def export_results_as_html(self):
        img_with_overlay = self.overlay_contours()
        path_to_image = self.save_image("full_overlay", img_with_overlay)

        # Export one processes page as html for further review
        tpl = env.get_template("page.html")

        shreds = self.get_shreds()
        for c in shreds:
            # Slight pre-processing of the features of each piece
            c.features["on_sheet_angle"] = "%8.1f&deg;" % c.features["on_sheet_angle"]
            c.features["ratio"] = "%8.2f" % c.features["ratio"]
            c.features["solidity"] = "%8.2f" % c.features["solidity"]

        export_dir, img_name = os.path.split(path_to_image)

        with open("%s/index.html" % export_dir, "w") as fp:
            fp.write(tpl.render(
                img_name=img_name,
                contours=shreds,
                export_dir=export_dir,
                out_dir_name=self.sheet_name,
            ))

    def save_thumb(self, width=200):
        r = float(width) / self.sheet.orig_img.shape[1]
        dim = (width, int(self.sheet.orig_img.shape[0] * r))

        resized = cv2.resize(self.sheet.orig_img, dim,
                             interpolation=cv2.INTER_AREA)
        return self.save_image("thumb", resized)

    def get_shreds(self):
        return self.sheet.get_shreds(self.feature_extractors, self.sheet_name)


if __name__ == '__main__':
    args = parser.parse_args()

    fnames = args.fnames
    out_format = args.out_format

    out_dir = "../out"

    static_dir = os.path.join(out_dir, "static")
    if os.path.exists(static_dir):
        shutil.rmtree(static_dir)
    shutil.copytree("static", static_dir)

    sheets = []
    # Here we are processing all files one by one and also generating
    # index sheet for it
    for fname in glob(fnames):
        sheet_name = os.path.splitext(os.path.basename(fname))[0]

        print("Processing file %s" % fname)
        sheet = SheetIO(fname, sheet_name,
                        [GeometryFeatures, ColourFeatures, LinesFeatures],
                        out_dir, out_format)

        sheet.export_results_as_html()
        sheets.append({
            "thumb": sheet.save_thumb(),
            "name": sheet.sheet_name
        })

    with open("../out/index.html", "w") as fp:
        tpl = env.get_template("index_sheet.html")

        fp.write(tpl.render(
            sheets=sheets
        ))
