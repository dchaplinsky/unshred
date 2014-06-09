import sys
import cv2
import numpy as np
import math


def open_image_and_separate_bg(fname):
    img = cv2.imread(fname)

    # Let's build a simple mask to separate pieces from background
    # just by checking if pixel is in range of some colours
    mask = cv2.inRange(img,
                       np.array([0, 0, 165], dtype=img.dtype),
                       np.array([200, 100, 255], dtype=img.dtype))
    mask = cv2.bitwise_not(mask)

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

    # Add some borders (just in case)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                             value=[0, 0, 0])
    mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])

    # Write original image with no background for debug purposes
    cv2.imwrite("debug/mask.tif", mask)

    return img, mask


def extract_piece_and_features(img, c, name):
    height, width, channels = img.shape

    # bounding rect of currrent contour
    r_x, r_y, r_w, r_h = cv2.boundingRect(c)
    area = cv2.contourArea(c)

    # filter out too small fragments
    if r_w <= 10 or r_h <= 10 or area < 100:
        print("Skipping piece #%s as too small" % name)
        return

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
    cv2.imwrite("pieces/%s.tif" % name, img_roi)

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
    corners = np.int0(corners)

    # Draw contours for debug purposes
    mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGR)
    cv2.drawContours(mask, contours, -1, (255, 0, 255), 2)

    # And put corners on top
    for i in corners:
        x, y = i.ravel()
        cv2.circle(mask, (x, y), 3, [0, 255, 0], -1)

    cnt = contours[0]

    # Let's find features that will help us to determine top side of the piece
    if mask.shape[0] / float(mask.shape[1]) >= 2.7:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        # Check for convex defects to see if we can find a trace of a
        # shredder cut which is usually on top of the piece.
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
                        cv2.circle(mask, far, 5, [0, 0, 255], -1)

        # Also top and bottom points on the contour
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        cv2.circle(mask, topmost, 3, [0, 255, 255], -1)
        cv2.circle(mask, bottommost, 3, [0, 255, 255], -1)

    cv2.imwrite("pieces/%s_mask.tif" % name, mask)


if __name__ == '__main__':
    # Open an image here
    fname = "src/puzzle_small.tif" if len(sys.argv) == 1 else sys.argv[1]

    img, mask = open_image_and_separate_bg(fname)

    # Find contours of pieces
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Walk through contrours to extract pieces and unify the rotation
    for i, c in enumerate(contours):
        extract_piece_and_features(img, c, i)

    # Another useful debug: drawing contours and their min area boxes
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    boxes = map(cv2.minAreaRect, contours)
    boxes2draw = map(lambda b: np.int0(cv2.cv.BoxPoints(b)), boxes)
    cv2.drawContours(img, boxes2draw, -1, (0, 0, 255), 2)
    cv2.imwrite("debug/out_with_contours.tif", img)
