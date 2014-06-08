import sys
import cv2
import numpy as np
import math

# Open an image here
fname = "src/puzzle_small.tif" if len(sys.argv) == 1 else sys.argv[1]
img = cv2.imread(fname)

# Let's build a simple mask to separate pieces from background
# just by checking if pixel is in range of some colours
mask = cv2.inRange(img,
                   np.array([0, 0, 190], dtype=img.dtype),
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

# Find contours of pieces
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

width, height, channels = img.shape

# Walk through contrours to extract pieces and unify the rotation
for i, c in enumerate(contours):
    # bounding rect of currrent contour
    r_x, r_y, r_w, r_h = cv2.boundingRect(c)

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
    y2 = math.ceil(box_center[1] + bbox[1] / 2)
    x1 = math.floor(box_center[0] - bbox[0] / 2)
    x2 = math.ceil(box_center[0] + bbox[0] / 2)

    # A mask we use to show only piece we are currently working on
    mask = np.zeros([width, height, 1], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, cv2.cv.CV_FILLED)

    # applying mask to original image
    img_roi = cv2.bitwise_and(img, img, mask=mask)

    # Rotating it
    M = cv2.getRotationMatrix2D(box_center, angle, 1)
    # Croping it
    img_roi = cv2.warpAffine(img_roi, M, (height, width))[y1:y2, x1:x2]

    cv2.imwrite("pieces/%s.tif" % i, img_roi)

# Another useful debug: drawing contours and their min area boxes
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
boxes = map(cv2.minAreaRect, contours)
boxes2draw = map(lambda b: np.int0(cv2.cv.BoxPoints(b)), boxes)
cv2.drawContours(img, boxes2draw, -1, (0, 0, 255), 2)
cv2.imwrite("debug/out_with_contours.tif", img)
