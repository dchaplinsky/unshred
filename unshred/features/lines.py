import cv2
import numpy
from unshred.features import AbstractShredFeature


class LinesFeatures(AbstractShredFeature):
    TAG_HAS_LINES_FEATURE = "has lines"
    TAG_PARALLEL_FEATURE = "parallel"
    TAG_PERPENDECULAR_FEATURE = "perpendecular"

    def get_info(self, shred, contour, name):

        tags = []
        params = {}

        _, _, _, mask = cv2.split(shred)
        #
        # # expanding mask for future removal of a border
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, (3, 3), iterations=2)
        #
        # # thresholding our shred
        gray = cv2.cvtColor(shred, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 5, 1)
        # attemting to remove borders
        edges = edges & mask
        # reducing noise
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, (6, 6), iterations=2)
        # removing small white noise
        edges = cv2.medianBlur(edges, 3)
        # const: uncomment for debug
        cv2.imwrite('../debug/edges_%s.png' % name, edges)

        lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 20, minLineLength=50, maxLineGap=30)
        if not lines is None:
            # const: uncomment for debug
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(shred, (x1, y1), (x2, y2), (255, 255, 0, 0), 2)
            cv2.imwrite('../debug/houghlines_%s.png' % name, shred)

            params['Lines Count'] = len(lines[0])
            tags.append(self.TAG_HAS_LINES_FEATURE)

        return params, tags
