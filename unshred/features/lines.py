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
        # gray = cv2.blur(gray, (5,5))
        # thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thimg = Image.fromarray(thresh[1])
        # thimg.save('../debug/thresh_%s.png'%name)
        edges = cv2.Canny(gray, 100, 200, apertureSize = 3)
        # removing contours from the edges (by drawing them black)
        cv2.drawContours(edges, contour, -1, (0, 0, 0), 18)
        cv2.imwrite('../debug/edges_%s.png'%name, edges)

        lines = cv2.HoughLines(edges, 1, 1* numpy.pi/180, 40)

        if not lines is None:
            ar = lines[0]
            # sorting by theta (for grouping by angle)
            ar = ar[ar[:,1].argsort()]
            #debug images
            for rho,theta in ar:
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b)) # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
                y1 = int(y0 + 1000*(a)) # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
                x2 = int(x0 - 1000*(-b)) # But we need integers, so use int() function after that, ie int(np.around(x))
                y2 = int(y0 - 1000*(a))
                cv2.line(gray,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.imwrite('../debug/houghlines_%s.png'%name, gray)

            params['Lines Count'] = len(lines)
            tags.append(self.TAG_HAS_LINES_FEATURE)

        return params, tags
