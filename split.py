import os
import os.path
import shutil
import sys
import cv2
import numpy as np
import math
from glob import glob
from jinja2 import FileSystemLoader, Environment

out_format = "png"
# TODO: refactor to avoid using global variable
out_dir_name = "test"

env = Environment(loader=FileSystemLoader("templates"))


def save_image(fname, img, format=None):
    full_out_dir, fname = os.path.split(
        os.path.join("out", out_dir_name, fname))

    try:
        os.makedirs(full_out_dir)
    except OSError:
        pass

    if format is None:
        format = out_format

    fname = "%s/%s.%s" % (full_out_dir, fname, format)
    cv2.imwrite(fname, img)

    return fname


backgrounds = [
    # DARPA SHRED task #1
    [[np.array([0, 0, 165]), np.array([200, 100, 255])]],
    # Pink
    [[np.array([160, 120, 230]), np.array([200, 210, 255])],
     [np.array([210, 185, 245]), np.array([235, 195, 255])],
     ],
]


def replace_scanner_background(img):
    # Here we are trying to find scanner background
    # (gray borders around colored sheet where shreds are glued)
    # Problem here is that colored sheet might have borders of different sizes
    # on different sides of it and we don't know the color of the sheet.
    # Also sheets isn't always has perfectly straight edges or rotated slightly
    # against edges of the scanner
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
    fimg = cv2.morphologyEx(fimg, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8),
                            iterations=2)
    fimg = fimg[5:-5, 5:-5]

    # Searching for a biggest outter contour
    contours, _ = cv2.findContours(cv2.bitwise_not(fimg), cv2.RETR_EXTERNAL,
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


def find_appropriate_mask(img):
    # Let's build a simple mask to separate pieces from background
    # just by checking if pixel is in range of some colours
    m = 0
    res = None

    # Here we calculate mask to separate background of the scanner
    scanner_bg = replace_scanner_background(img)

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

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # and run grabcut algo against the seed mask to refine background
    # separation
    # cv2.imwrite("debug/mask1.png", res)

    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((3, 7), np.uint8))
    cv2.imwrite("debug/mask2.png", res)

    # res = cv2.morphologyEx(res, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    # cv2.imwrite("debug/mask3.png", res)

    # mask = np.where((res == 255), cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype('uint8')
    # cv2.grabCut(hsv, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    # res = np.where((mask == 2) | (mask == 0), 255, 0).astype('uint8')

    cv2.imwrite("debug/mask4.png", res)
    return cv2.bitwise_not(res)


def open_image_and_separate_bg(img):
    mask = find_appropriate_mask(img)

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


def extract_piece_and_features(img, c, name):
    height, width, channels = img.shape

    # bounding rect of currrent contour
    r_x, r_y, r_w, r_h = cv2.boundingRect(c)
    area = cv2.contourArea(c)

    # filter out too small fragments
    if r_w <= 20 or r_h <= 20 or area < 200:
        print("Skipping piece #%s as too small" % name)
        return None

    # Generating simplified contour to use it in html
    epsilon = 0.01 * cv2.arcLength(c, True)
    simplified_contour = cv2.approxPolyDP(c, epsilon, True)

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    if solidity < 0.75:
        print("Piece #%s looks suspicious" % name)

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
    img_crp = img[r_y:r_y + r_h, r_x:r_x + r_w]
    piece_in_context = save_image(
        "pieces/%s_ctx" % name,
        img[r_y - 10:r_y + r_h + 10, r_x - 10:r_x + r_w + 10])

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
    piece_fname = save_image("pieces/%s" % name, img_roi, "png")

    # FEATURES MAGIC BELOW
    #
    # Get our mask/contour back after the trasnform
    _, _, _, mask = cv2.split(img_roi)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        print("Piece #%s has strange contours after transform" % name)

    # Let's also detect 25 of most outstanding corners of the contour.
    corners = cv2.goodFeaturesToTrack(mask, 25, 0.01, 10)
    corners = np.int0(corners) if corners is not None else []

    # edges = cv2.Canny(img_roi[:, :, 2], 100, 200)
    # save_image("pieces/%s_edges" % name, edges)

    # Draw contours for debug purposes
    mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGRA)
    mask[:, :, 3] = mask[:, :, 0]
    cv2.drawContours(mask, contours, -1, (255, 0, 255, 255), 2)

    # And put corners on top
    for i in corners:
        x, y = i.ravel()
        cv2.circle(mask, (x, y), 3, (0, 255, 0, 255), -1)

    cnt = contours[0]

    # Let's find features that will help us to determine top side of the piece
    if mask.shape[0] / float(mask.shape[1]) >= 2.7:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        # Check for convex defects to see if we can find a trace of a
        # shredder cut which is usually on top of the piece.

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = tuple(cnt[f][0])

                # if convex defect is big enough
                if d / 256. > 7:
                    # And lays at the top or bottom of the piece
                    y_dist = min(abs(0 - far[1]), abs(mask.shape[0] - far[1]))
                    if float(y_dist) / mask.shape[0] < 0.1:
                        # and more or less is in the center
                        if abs(far[0] - mask.shape[1] / 2.) / mask.shape[1] < 0.25:
                            cv2.circle(mask, far, 5, [0, 0, 255, 255], -1)

    # Also top and bottom points on the contour
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    cv2.circle(mask, topmost, 3, [0, 255, 255, 255], -1)
    cv2.circle(mask, bottommost, 3, [0, 255, 255, 255], -1)

    features_fname = save_image("pieces/%s_mask" % name, mask, "png")
    return {
        "features": {
            "area": area,
            "angle": angle,
            "ratio": mask.shape[0] / float(mask.shape[1]),
            "solidity": solidity,
            "x": r_x,
            "y": r_y,
            "width": r_w,
            "height": r_h
        },
        "contour": c,
        "name": name,
        "piece_fname": piece_fname,
        "features_fname": features_fname,
        "piece_in_context_fname": piece_in_context,
        "simplified_contour": simplified_contour,
    }


def overlay_contours(img, contours):
    # Draw contours on top of image with a nice yellow tint
    overlay = np.zeros(img.shape, np.uint8)
    contours = map(lambda x: x["contour"], contours)

    cv2.fillPoly(overlay, contours, [104, 255, 255])
    img = img.copy() + overlay

    cv2.drawContours(img, contours, -1, [0, 180, 0], 2)
    return img


def convert_poly_to_string(poly):
    # Helper to convert openCV contour to a string with coordinates
    # that is recognizible by html area tag
    # TODO: Jinja2 filter
    return ",".join(map(lambda x: ",".join(map(str, x[0])), poly))


def export_results_as_html(path_to_image, contours):
    # Export one processes page as html for further review
    tpl = env.get_template("page.html")

    for c in contours:
        c["simplified_contour"] = convert_poly_to_string(
            c["simplified_contour"])

        # Slight pre-processing of the features of each piece
        c["features"]["angle"] = "%8.1f&deg;" % c["features"]["angle"]
        c["features"]["ratio"] = "%8.2f" % c["features"]["ratio"]
        c["features"]["solidity"] = "%8.2f" % c["features"]["solidity"]

    export_dir, img_name = os.path.split(path_to_image)

    with open("%s/index.html" % export_dir, "w") as fp:
        fp.write(tpl.render(
            img_name=img_name,
            contours=contours,
            export_dir=export_dir,
            out_dir_name=out_dir_name
        ))


def process_file(fname):
    # Process single file
    orig_img = cv2.imread(fname)

    processed_img, mask = open_image_and_separate_bg(orig_img)

    # Find contours of pieces
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Walk through contours to extract pieces and unify the rotation
    resulting_contours = []
    for i, c in enumerate(contours):
        cnt_features = extract_piece_and_features(orig_img, c, i)
        if cnt_features is not None:
            resulting_contours.append(cnt_features)

    return orig_img, resulting_contours


if __name__ == '__main__':
    fnames = "src/puzzle_small.tif" if len(sys.argv) == 1 else sys.argv[1]
    out_format = "png" if len(sys.argv) == 2 else sys.argv[2]

    static_dir = os.path.join("out/static")
    if os.path.exists(static_dir):
        shutil.rmtree(static_dir)
    shutil.copytree("static", static_dir)

    sheets = []
    # Here we are processing all files one by one and also generating
    # index sheet for it
    for fname in glob(fnames):
        out_dir_name = os.path.splitext(os.path.basename(fname))[0]

        print("Processing file %s" % fname)
        orig_img, resulting_contours = process_file(fname)

        img_with_overlay = overlay_contours(orig_img, resulting_contours)
        path_to_image = save_image("full_overlay", img_with_overlay)

        export_results_as_html(path_to_image, resulting_contours)

        r = 200.0 / orig_img.shape[1]
        dim = (200, int(orig_img.shape[0] * r))

        resized = cv2.resize(orig_img, dim, interpolation=cv2.INTER_AREA)
        path_to_image = save_image("thumb", resized)

        sheets.append({
            "name": out_dir_name,
            "thumb": path_to_image
        })

    with open("out/index.html", "w") as fp:
        tpl = env.get_template("index_sheet.html")

        fp.write(tpl.render(
            sheets=sheets
        ))
