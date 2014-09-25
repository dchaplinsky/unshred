"""Fancy adaptive threshlding.

Code adapted from
http://stackoverflow.com/questions/22122309/opencv-adaptive-threshold-ocr.

As I understand it:
1. Reduces an image to a smaller one, where each
DEFAULT_BLOCKSIZExDEFAULT_BLOCKSIZE block -> one pixel.
2. Creates a mask of small_image size, where pixels corresponding to
high-variance (non-background) blocks get value > 0, others (background
blocks) get 0.
3. Small image is inpainted using mask from step 2. So non-bg blocks are
inpainted by surrounding bg blocks.
4. Image is resize back to original size, resulting in what looks like just bg
from original image.
5. Bg image is subtracted from original and result thresholded.

The algorith assumes dark foreground on light background.

"""
import cv2
import numpy as np

DEFAULT_BLOCKSIZE = 40

# Blocks with variance over this value are assumed to contain foreground.
MEAN_VARIANCE_THRESHOLD = 0.01


def _calc_block_mean_variance(image, mask, blocksize):
    """Adaptively determines image background.

    Args:
        image: image converted 1-channel image.
        mask: 1-channel mask, same size as image.
        blocksize: adaptive algorithm parameter.

    Returns:
        image of same size as input with foreground inpainted with background.
    """
    I = image.copy()
    I_f = I.astype(np.float32) / 255.  # Used for mean and std.

    result = np.zeros(
        (image.shape[0] / blocksize, image.shape[1] / blocksize),
        dtype=np.float32)

    for i in xrange(0, image.shape[0] - blocksize, blocksize):
        for j in xrange(0, image.shape[1] - blocksize, blocksize):

            patch = I_f[i:i+blocksize+1, j:j+blocksize+1]
            mask_patch = mask[i:i+blocksize+1, j:j+blocksize+1]

            tmp1 = np.zeros((blocksize, blocksize))
            tmp2 = np.zeros((blocksize, blocksize))
            mean, std_dev = cv2.meanStdDev(patch, tmp1, tmp2, mask_patch)

            value = 0
            if std_dev[0][0] > MEAN_VARIANCE_THRESHOLD:
                value = mean[0][0]

            result[i/blocksize, j/blocksize] = value

    small_image = cv2.resize(I, (image.shape[1] / blocksize,
                                 image.shape[0] / blocksize))

    res, inpaintmask = cv2.threshold(result, 0.02, 1, cv2.THRESH_BINARY)

    inpainted = cv2.inpaint(small_image, inpaintmask.astype(np.uint8), 5,
                            cv2.INPAINT_TELEA)

    res = cv2.resize(inpainted, (image.shape[1], image.shape[0]))

    return res


def threshold(image, block_size=DEFAULT_BLOCKSIZE, mask=None):
    """Applies adaptive thresholding to the given image.

    Args:
        image: BGRA image.
        block_size: optional int block_size to use for adaptive thresholding.
        mask: optional mask.
    Returns:
        Thresholded image.
    """
    if mask is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[:] = 255

    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    res = _calc_block_mean_variance(image, mask, block_size)
    res = image.astype(np.float32) - res.astype(np.float32) + 255
    _, res = cv2.threshold(res, 215, 255, cv2.THRESH_BINARY)
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file name.',
                        nargs='?', default="11.jpg")
    parser.add_argument('output', type=str, help='Output file name.',
                        nargs='?', default="out.png")

    args = parser.parse_args()

    fname = args.input
    outfile = args.output

    image = cv2.imread(fname, cv2.CV_LOAD_IMAGE_UNCHANGED)
    result = threshold(image)
    cv2.imwrite(outfile, result * 255)
