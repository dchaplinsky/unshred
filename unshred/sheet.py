import collections
import cv2
import math
import numpy as np


class Shred(collections.namedtuple('_Shred',
            'contour features name piece_fname features_fname '
            'piece_in_context_fname simplified_contour sheet img_roi '
            'tags_suggestions')):
    """Stores the shred information, incl. tags suggestions."""
    def __repr__(self):
        return '<Shred #%s of sheet %s>' % (self.name, self.sheet)


class Sheet(object):
    """Represents a single sheet with shred extraction methods.

    Public API includes initializer and get_shreds() method.
    """

    _backgrounds = [
        # DARPA SHRED task #1
        [
            [np.array([30, 190, 180]), np.array([180, 255, 255])],
            [np.array([0, 240, 170]), np.array([50, 255, 255])]
        ],
        # Pink
        [[np.array([160, 50, 210]), np.array([200, 150, 255])]],

        # Denys $100 paper background
        [
            [np.array([0, 100, 200]), np.array([20, 255, 255])],
            [np.array([140, 120, 200]), np.array([250, 255, 255])],
        ],

        # AB's brown paper
        [
            [np.array([0, 70, 120]), np.array([30, 140, 200])],
        ]
    ]

    def __init__(self, orig_image, dpi, save_image):
        """Initializes a Sheet instance.

        Args:
            orig_image: cv.Mat instance with the original sheet image.
            dpi: optional (x resolution, y resolution) tuple or None.
                If set to None, will try to guess dpi.
            save_image: A callback to save debug images with args (name, img)
        """

        self._shreds = None
        self.orig_img = orig_image

        self.save_image = save_image

        self._fg_mask = None
        self._shreds = None

        if dpi is None:
            self.res_x, self.res_y = self._guess_dpi()
        else:
            self.res_x, self.res_y = dpi

    def px_to_mm(self, px):
        """Convert given value in px to a value in millimetres according to
        sheet's DPI

        Args:
            px: integer, value in pixels

        Returns:
            value in millimetres
        """
        return float(px) / self.res_x * 25.4

    def get_shreds(self, feature_extractors, sheet_name):
        """Detects shreds in the current sheet and constructs Shred instances.

        Caches the results for further invocations.

        Args:
            feature_extractors: iterable of AbstractShredFeature instances to
                use for shreds feature assignment.
            sheet_name: string, included in shred attributes.

        Returns:
            list of Shred instances.
        """
        if self._shreds is None:
            shreds = []
            contours, _ = cv2.findContours(self._foreground_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                shred = self._make_shred(contour, i, feature_extractors,
                                         sheet_name)
                if shred is not None:
                    shreds.append(shred)
            self._shreds = shreds
        return self._shreds

    def _guess_dpi(self):
        # Note: this is inconsistent for images with larger y dimension:
        # if they have exif, dpi is returned for (x, y), otherwise it's (y, x).

        # Let's assume that we are dealing with A4 (8.27 in x 11.7) and try to
        # guess it
        w, h = min(self.orig_img.shape[:2]), max(self.orig_img.shape[:2])
        xres, yres = w / 8.27, h / 11.7

        # Will suffice for now
        if max(xres, yres) / min(xres, yres) > 1.1:
            raise ValueError("Dpi is not provided and can't be guessed.")

        return int(round(xres, -2)), int(round(yres, -2))

    def _detect_scanner_background(self, img):
        # Method returns a mask describing detected scanner background
        # (gray borders around colored sheet where shreds are glued).
        # Problem here is that colored sheet might have borders of different
        # sizes on different sides of it and we don't know the color of the
        # sheet.
        # Also sheets don't always have perfectly straight edges or are
        # slightly rotated against edges of the scanner.
        # Idea is relatively simple:

        # Convert image to LAB and grab A and B channels.
        fimg = cv2.cvtColor(img, cv2.cv.CV_BGR2Lab)
        _, a_channel, b_channel = cv2.split(fimg)

        def try_method(fimg, border, aggressive=True):
            fimg = cv2.copyMakeBorder(fimg, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                      value=border)

            # Flood fill supposed background with white (255).
            if aggressive:
                cv2.floodFill(fimg, None, (10, 10), 255, 1, 1)
            else:
                cv2.floodFill(fimg, None, (10, 10), 255, 2, 2,
                              cv2.FLOODFILL_FIXED_RANGE)

            # Binarize image into white background and black everything else.
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
        # and flood fill scanner background starting from most
        # aggressive flood fill settings to least aggressive.
        for i, opt in enumerate(options):
            fimg, hist = try_method(*opt)
            # First setting that doesn't flood that doesn't hurt colored sheet
            # too badly wins.
            bg_ratio = hist[1] / sum(hist)
            if bg_ratio < 0.2:
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

        # Cannot say I'm satisfied with algo but using grabcut here seems too
        # expensive. Also it was a lot of fun to build it.
        return fimg

    def _find_foreground_mask(self):
        # Let's build a simple mask to separate pieces from background
        # just by checking if pixel is in range of some colours
        m = 0
        res = None

        # Here we calculate mask to separate background of the scanner
        scanner_bg = self._detect_scanner_background(self.orig_img)

        # And here we are trying to check different ranges for different
        # background to find the winner.
        hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)

        for bg in self._backgrounds:
            mask = np.zeros(self.orig_img.shape[:2], np.uint8)

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

        res = cv2.bitwise_not(res)
        # Init kernel for dilate/erode
        kernel = np.ones((5, 5), np.uint8)
        # optional blur of mask
        # mask = cv2.medianBlur(mask, 5)

        # Clean noise on background
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        # Clean noise inside pieces
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

        # Write original image with no background for debug purposes
        #cv2.imwrite("debug/mask.tif", mask)
        return res

    @property
    def _foreground_mask(self):
        if self._fg_mask is None:
            self._fg_mask = self._find_foreground_mask()
        return self._fg_mask

    def _make_shred(self, c, name, feature_extractors, sheet_name):
        """Creates a Shred instances from a given contour.

        Args:
            c: cv2 contour object.
            name: string shred name within a sheet.
            feature_extractors: iterable of AbstractShredFeature instances.

        Returns:
            A new Shred instance or None on failure.
        """
        height, width, channels = self.orig_img.shape

        # bounding rect of currrent contour
        r_x, r_y, r_w, r_h = cv2.boundingRect(c)

        # Generating simplified contour to use it in html
        epsilon = 0.01 * cv2.arcLength(c, True)
        simplified_contour = cv2.approxPolyDP(c, epsilon, True)

        # filter out too small fragments
        if self.px_to_mm(r_w) <= 3 or self.px_to_mm(r_h) <= 3:
            print("Skipping piece #%s as too small (%spx x %s px)" % (
                name, r_w, r_h))
            return None

        if self.px_to_mm(r_w) >= 100 and self.px_to_mm(r_h) >= 100:
            print("Skipping piece #%s as too big (%spx x %s px)" % (
                name, r_w, r_h))
            return None

        # position of rect of min area.
        # this will provide us angle to straighten image
        box_center, bbox, angle = cv2.minAreaRect(c)

        # We want our pieces to be "vertical"
        if bbox[0] > bbox[1]:
            angle += 90
            bbox = (bbox[1], bbox[0])

        if bbox[1] / float(bbox[0]) > 70:
            print("Skipping piece #%s as too too long and narrow" % name)
            return None

        # Coords of region of interest using which we should crop piece after
        # rotation
        y1 = math.floor(box_center[1] - bbox[1] / 2)
        x1 = math.floor(box_center[0] - bbox[0] / 2)
        bbox = tuple(map(int, map(math.ceil, bbox)))

        # A mask we use to show only piece we are currently working on
        piece_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv2.drawContours(piece_mask, [c], -1, 255, cv2.cv.CV_FILLED)

        # apply mask to original image
        img_crp = self.orig_img[r_y:r_y + r_h, r_x:r_x + r_w]
        piece_in_context = self.save_image(
            "pieces/%s_ctx" % name,
            self.orig_img[max(r_y - 10, 0):r_y + r_h + 10,
                          max(r_x - 10, 0):r_x + r_w + 10])

        mask = piece_mask[r_y:r_y + r_h, r_x:r_x + r_w]
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
            # On_sheet_* features describe the min counding box on the sheet.
            "on_sheet_x": r_x,
            "on_sheet_y": r_y,
            "on_sheet_width": r_w,
            "on_sheet_height": r_h,
            "on_sheet_angle": angle,
            "width": img_roi.shape[1],
            "height": img_roi.shape[0],
        }

        tags_suggestions = []
        for feat in feature_extractors:
            fts, tags = feat.get_info(img_roi, cnt, name)
            base_features.update(fts)
            tags_suggestions += tags

        if tags_suggestions:
            print(name, tags_suggestions)

        return Shred(
            contour=c,
            features=base_features,
            features_fname=features_fname,
            img_roi=img_roi,
            name=name,
            piece_fname=piece_fname,
            piece_in_context_fname=piece_in_context,
            sheet=sheet_name,
            simplified_contour=simplified_contour,
            tags_suggestions=tags_suggestions,
        )
