import cv2
import numpy

from unshred import threshold
from unshred.features import AbstractShredFeature


DEBUG = False


class LinesFeatures(AbstractShredFeature):
    TAG_HAS_LINES_FEATURE = "Has Lines"

    def get_info(self, shred, contour, name):
        tags = []
        params = {}

        _, _, _, mask = cv2.split(shred)
        #
        # # expanding mask for future removal of a border
        kernel = numpy.ones((5, 5), numpy.uint8)
        if DEBUG:
            cv2.imwrite('../debug/%s_mask_0.png' % name, mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=2)
        if DEBUG:
            cv2.imwrite('../debug/%s_mask_1.png' % name, mask)
        _, mask = cv2.threshold(mask, 240, 0, cv2.THRESH_TOZERO)
        if DEBUG:
            cv2.imwrite('../debug/%s_mask_2.png' % name, mask)

        # TODO: move thresholding to Shred class, to allow reusing from other
        # feature detectors.
        edges = 255 - threshold.threshold(
            shred, min(shred.shape[:2])).astype(numpy.uint8)
        edges = edges & mask

        if DEBUG:
            # const: uncomment for debug
            cv2.imwrite('../debug/%s_asrc.png' % name, shred)
            cv2.imwrite('../debug/%s_edges.png' % name, edges)
            cv2.imwrite('../debug/%s_mask.png' % name, mask)

        _, _, r_w, r_h = cv2.boundingRect(contour)

        # Line len should be at least 50% of shred's width, gap - 20%
        # TODO: come up with better threshold value. Come up with better lines
        # filtering.
        lines = cv2.HoughLinesP(edges, rho=1, theta=numpy.pi / 180,
                                threshold=30, minLineLength=r_w * 0.5,
                                maxLineGap=r_w * 0.2)
        if lines is not None:
            if DEBUG:
                for x1, y1, x2, y2 in lines[0]:
                    cv2.line(shred, (x1, y1), (x2, y2), (255, 255, 0, 0), 2)
                cv2.imwrite('../debug/%s_houghlines.png' % name, shred)
            tags.append(self.TAG_HAS_LINES_FEATURE)

            # TODO: Find dominant lines slopes and store as a tag.
            # Determine the presence of multiple lines parallel or orthogonal to
            # dominant slope (might mean that's part of a table).

        return params, tags
