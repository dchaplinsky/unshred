import os
import os.path
import shutil
import sys
import cv2
import math
import numpy as np
from glob import glob
import exifread
from jinja2 import FileSystemLoader, Environment
from features import GeometryFeatures, ColourFeatures


def convert_poly_to_string(poly):
    # Helper to convert openCV contour to a string with coordinates
    # that is recognizible by html area tag
    return ",".join(map(lambda x: ",".join(map(str, x[0])), poly))

env = Environment(loader=FileSystemLoader("templates"))
env.filters["convert_poly_to_string"] = convert_poly_to_string


class Sheet(object):
    backgrounds = [
        # DARPA SHRED task #1
        [[np.array([0, 0, 165]), np.array([200, 100, 255])]],
        # Pink
        [[np.array([160, 120, 230]), np.array([200, 210, 255])],
         [np.array([210, 185, 245]), np.array([235, 195, 255])],
         ],
    ]

    def __init__(self, fname, sheet_name, feature_extractors,
                 out_dir="out", out_format="png"):

        self.fname = fname
        self.sheet_name = sheet_name
        self.out_format = out_format
        self.out_dir = out_dir

        self.orig_img = cv2.imread(fname)
        self.res_x, self.res_y = self.determine_dpi()
        self.feature_extractors = [feat(self) for feat in feature_extractors]

        processed_img, mask = self.open_image_and_separate_bg(self.orig_img)

        # Find contours of pieces
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Walk through contours to extract pieces and unify the rotation
        self.resulting_contours = []
        for i, c in enumerate(contours):
            cnt_features = self.extract_piece_and_features(c, i)
            if cnt_features is not None:
                self.resulting_contours.append(cnt_features)

    def determine_dpi(self):
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

        try:
            if "Image XResolution" in tags and "Image YResolution" in tags:
                return (parse_resolution(tags["Image XResolution"]),
                        parse_resolution(tags["Image YResolution"]))
        except ValueError:
            pass

        # Resolution cannot be determined via exif, let's assume that we are
        # dealing with A4 (8.27 in x 11.7) and try to guess it

        w, h = min(self.orig_img.shape[:2]), max(self.orig_img.shape[:2])
        xres, yres = w / 8.27, h / 11.7

        # Will suffice for now
        if max(xres, yres) / min(xres, yres) > 1.1:
            raise ValueError

        return int(round(xres, -2)), int(round(yres, -2))

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

    def replace_scanner_background(self, img):
        # Here we are trying to find scanner background
        # (gray borders around colored sheet where shreds are glued)
        # Problem here is that colored sheet might have borders of different
        # sizes on different sides of it and we don't know the color of the
        # sheet.
        # Also sheets isn't always has perfectly straight edges or rotated
        # slightly against edges of the scanner
        # Idea is relatively simple:

        # Convert image to LAB and grab A and B channels.
        fimg = cv2.cvtColor(img, cv2.cv.CV_BGR2Lab)
        _, a_channel, b_channel = cv2.split(fimg)

        def try_method(fimg, border, aggressive=True):
            fimg = cv2.copyMakeBorder(fimg, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                      value=border)

            if aggressive:
                cv2.floodFill(fimg, None, (10, 10), 255, 1, 1)
            else:
                cv2.floodFill(fimg, None, (10, 10), 255, 2, 2,
                              cv2.FLOODFILL_FIXED_RANGE)

            _, fimg = cv2.threshold(fimg, 254, 255, cv2.THRESH_BINARY)

            hist = cv2.calcHist([fimg], [0], None, [2], [0, 256])
            return fimg, hist

        options = [
            [a_channel, 127],
            [b_channel, 134],
            [a_channel, 127, False],
            [b_channel, 134, False],
        ]

        # And then try to add a border of predefined color around each channel
        # and flood fill scanner scanner background starting from most
        # aggressive flood fill settings to least aggressive.
        for i, opt in enumerate(options):
            fimg, hist = try_method(*opt)
            # First setting that doesn't flood that doesn't hurt colored sheet
            # too badly wins.
            if hist[1] < (hist[0] + hist[1]) * 0.2:
                break

        # Then we dilate it a bit.
        fimg = cv2.morphologyEx(fimg, cv2.MORPH_DILATE,
                                np.ones((5, 5), np.uint8), iterations=2)
        fimg = fimg[5:-5, 5:-5]

        # Searching for a biggest outter contour
        contours, _ = cv2.findContours(cv2.bitwise_not(fimg),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        main_contour = np.array(map(cv2.contourArea, contours)).argmax()

        # build a convex hull for it
        hull = cv2.convexHull(contours[main_contour])

        # And make a mask out of it.
        fimg = np.zeros(fimg.shape[:2], np.uint8)
        cv2.fillPoly(fimg, [hull], 255)
        fimg = cv2.bitwise_not(fimg)

        # Done. Piece of cake.
        return fimg
        # Cannot say I'm satisfied with algo but using grabcut here seems too
        # expensive. Also it was a lot of fun to build it.

    def find_appropriate_mask(self, img):
        # Let's build a simple mask to separate pieces from background
        # just by checking if pixel is in range of some colours
        m = 0
        res = None

        # Here we calculate mask to separate background of the scanner
        scanner_bg = self.replace_scanner_background(img)

        backgrounds = [[
            # [np.array([160, 35, 210]), np.array([180, 150, 255])],
            [np.array([160, 50, 210]), np.array([200, 150, 255])]
        ]]

        # And here we are trying to check different ranges for different
        # background to find the winner.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for bg in backgrounds:
            mask = np.zeros(img.shape[:2], np.uint8)

            for rng in bg:
                # As each pre-defined background is described by a list of
                # color ranges we are summing up their masks
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, rng[0], rng[1]))

            hist = cv2.calcHist([mask], [0], None, [2], [0, 256])
            # And here we are searching for a biggest possible mask across all
            # possible predefined backgrounds
            if hist[1] > m:
                m = hist[1]
                res = mask

        # then we remove scanner background
        res = cv2.bitwise_or(scanner_bg, res)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((3, 7), np.uint8))

        return cv2.bitwise_not(res)

    def open_image_and_separate_bg(self, img):
        mask = self.find_appropriate_mask(img)

        # Init kernel for dilate/erode
        kernel = np.ones((5, 5), np.uint8)
        # optional blur of mask
        # mask = cv2.medianBlur(mask, 5)

        # Clean noise on background
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Clean noise inside pieces
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to image
        img = cv2.bitwise_and(img, img, mask=mask)

        # Write original image with no background for debug purposes
        cv2.imwrite("debug/mask.tif", mask)

        return img, mask

    def extract_piece_and_features(self, c, name):
        height, width, channels = self.orig_img.shape

        # bounding rect of currrent contour
        r_x, r_y, r_w, r_h = cv2.boundingRect(c)

        # Generating simplified contour to use it in html
        epsilon = 0.01 * cv2.arcLength(c, True)
        simplified_contour = cv2.approxPolyDP(c, epsilon, True)

        area = cv2.contourArea(c)
        # filter out too small fragments
        if r_w <= 20 or r_h <= 20 or area < 500:
            print("Skipping piece #%s as too small" % name)
            return None

        # position of rect of min area.
        # this will provide us angle to straighten image
        box_center, bbox, angle = cv2.minAreaRect(c)

        # We want our pieces to be "vertical"
        if bbox[0] > bbox[1]:
            angle += 90
            bbox = (bbox[1], bbox[0])

        # Coords of region of interest using which we should crop piece after
        # rotation
        y1 = math.floor(box_center[1] - bbox[1] / 2)
        x1 = math.floor(box_center[0] - bbox[0] / 2)
        bbox = tuple(map(int, map(math.ceil, bbox)))

        # A mask we use to show only piece we are currently working on
        mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.cv.CV_FILLED)

        # apply mask to original image
        img_crp = self.orig_img[r_y:r_y + r_h, r_x:r_x + r_w]
        piece_in_context = self.save_image(
            "pieces/%s_ctx" % name,
            self.orig_img[max(r_y - 10, 0):r_y + r_h + 10,
                          max(r_x - 10, 0):r_x + r_w + 10])

        mask = mask[r_y:r_y + r_h, r_x:r_x + r_w]
        img_roi = cv2.bitwise_and(img_crp, img_crp, mask=mask)

        # Add alpha layer and set it to the mask
        img_roi = cv2.cvtColor(img_roi, cv2.cv.CV_BGR2BGRA)
        img_roi[:, :, 3] = mask[:, :, 0]

        # Straighten it
        # Because we crop original image before rotation we save us some memory
        # and a lot of time but we need to adjust coords of the center of
        # new min area rect
        M = cv2.getRotationMatrix2D((box_center[0] - r_x,
                                     box_center[1] - r_y), angle, 1)

        # And translate an image a bit to make it fit to the bbox again.
        # This is done with direct editing of the transform matrix.
        # (Wooohoo, I know matrix-fu)
        M[0][2] += r_x - x1
        M[1][2] += r_y - y1

        # Apply rotation/transform/crop
        img_roi = cv2.warpAffine(img_roi, M, bbox)
        piece_fname = self.save_image("pieces/%s" % name, img_roi, "png")

        # FEATURES MAGIC BELOW
        #
        # Get our mask/contour back after the trasnform
        _, _, _, mask = cv2.split(img_roi)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            print("Piece #%s has strange contours after transform" % name)

        cnt = contours[0]

        features_fname = self.save_image("pieces/%s_mask" % name, mask, "png")

        base_features = {
            "angle": angle,
            "pos_x": r_x,
            "pos_y": r_y,
            "pos_width": r_w,
            "pos_height": r_h
        }

        tags_suggestions = []

        for feat in self.feature_extractors:
            fts, tags = feat.get_info(img_roi, cnt, name)
            base_features.update(fts)
            tags_suggestions += tags

        if tags_suggestions:
            print(name, tags_suggestions)

        return {
            "features": base_features,
            "tags_suggestions": tags_suggestions,
            "contour": c,
            "name": name,
            "piece_fname": piece_fname,
            "features_fname": features_fname,
            "piece_in_context_fname": piece_in_context,
            "simplified_contour": simplified_contour,
            "sheet": self.sheet_name
        }

    def overlay_contours(self):
        # Draw contours on top of image with a nice yellow tint
        overlay = np.zeros(self.orig_img.shape, np.uint8)
        contours = map(lambda x: x["contour"], self.resulting_contours)

        cv2.fillPoly(overlay, contours, [104, 255, 255])
        img = self.orig_img.copy() + overlay

        cv2.drawContours(img, contours, -1, [0, 180, 0], 2)
        return img

    def export_results_as_html(self):
        img_with_overlay = self.overlay_contours()
        path_to_image = self.save_image("full_overlay", img_with_overlay)

        # Export one processes page as html for further review
        tpl = env.get_template("page.html")

        for c in self.resulting_contours:
            # Slight pre-processing of the features of each piece
            c["features"]["angle"] = "%8.1f&deg;" % c["features"]["angle"]
            c["features"]["ratio"] = "%8.2f" % c["features"]["ratio"]
            c["features"]["solidity"] = "%8.2f" % c["features"]["solidity"]

        export_dir, img_name = os.path.split(path_to_image)

        with open("%s/index.html" % export_dir, "w") as fp:
            fp.write(tpl.render(
                img_name=img_name,
                contours=self.resulting_contours,
                export_dir=export_dir,
                out_dir_name=self.sheet_name
            ))

    def save_thumb(self, width=200):
        r = float(width) / self.orig_img.shape[1]
        dim = (width, int(self.orig_img.shape[0] * r))

        resized = cv2.resize(self.orig_img, dim, interpolation=cv2.INTER_AREA)
        return self.save_image("thumb", resized)


if __name__ == '__main__':
    fnames = "src/puzzle_small.tif" if len(sys.argv) == 1 else sys.argv[1]
    out_format = "png" if len(sys.argv) == 2 else sys.argv[2]
    out_dir = "out"

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
        sheet = Sheet(fname, sheet_name,
                      [GeometryFeatures, ColourFeatures], out_dir, out_format)

        sheet.export_results_as_html()
        sheets.append({
            "thumb": sheet.save_thumb(),
            "name": sheet.sheet_name
        })

    with open("out/index.html", "w") as fp:
        tpl = env.get_template("index_sheet.html")

        fp.write(tpl.render(
            sheets=sheets
        ))
