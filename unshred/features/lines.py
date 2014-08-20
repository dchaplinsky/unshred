from PIL import Image
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

        gray = cv2.cvtColor(shred, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 5, 1)
        # removing contours from the edges (by drawing them black)
        cv2.drawContours(edges, contour, -1, (0, 0, 0), 24)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, (6, 6), iterations=2)

        # const: uncomment for debug
        # cv2.imwrite('../debug/edges_%s.png'%name, edges)

        lines = cv2.HoughLinesP(edges, 1, numpy.pi/180, 20, minLineLength = 30, maxLineGap = 10)
        if not lines is None:
            # const: uncomment for debug
            #debug images
            # for x1,y1,x2,y2 in lines[0]:
                # cv2.line(gray,(x1,y1),(x2,y2),(255,0,0),2)
            # cv2.imwrite('../debug/houghlines_%s.png'%name, gray)

            params['Lines Count'] = len(lines[0])
            tags.append(self.TAG_HAS_LINES_FEATURE)

        return params, tags
