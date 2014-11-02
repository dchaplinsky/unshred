import math
import cmath
import cv2
import numpy

from unshred import threshold
from unshred.features import AbstractShredFeature


DEBUG = False


MEAN, MEDIAN = range(2)


def _get_dominant_angle(lines, domination_type=MEDIAN):
    """Picks dominant angle of a set of lines.

    Args:
        lines: iterable of (x1, y1, x2, y2) tuples that define lines.
        domination_type: either MEDIAN or MEAN.

    Returns:
        Dominant angle value in radians.

    Raises:
        ValueError: on unknown domination_type.
    """
    if domination_type == MEDIAN:
        return _get_median_angle(lines)
    elif domination_type == MEAN:
        return _get_mean_angle(lines)
    else:
        raise ValueError('Unknown domination type provided: %s' % (
            domination_type))


def _normalize_angle(angle, range, step):
    """Finds an angle that matches the given one modulo step.

    Increments and decrements the given value with a given step.

    Args:
        range: a 2-tuple of min and max target values.
        step: tuning step.

    Returns:
        Normalized value within a given range.
    """
    while angle <= range[0]:
        angle += step
    while angle >= range[1]:
        angle -= step
    return angle


def _get_mean_angle(lines):
    unit_vectors = []
    for x1, y1, x2, y2 in lines:
        c = complex(x2, -y2) - complex(x1, -y1)
        unit = c / abs(c)
        unit_vectors.append(unit)

    avg_angle = cmath.phase(numpy.average(unit_vectors))

    return _normalize_angle(avg_angle, [-math.pi / 2, math.pi / 2], math.pi)


def _get_median_angle(lines):
    angles = []
    for x1, y1, x2, y2 in lines:
        c = complex(x2, -y2) - complex(x1, -y1)
        angle = cmath.phase(c)
        angles.append(angle)

    # Not np.median to avoid averaging middle elements.
    median_angle = numpy.percentile(angles, .5)

    return _normalize_angle(median_angle, [-math.pi / 2, math.pi / 2], math.pi)


class LinesFeatures(AbstractShredFeature):
    """Feature detector that recognizes lines.

    If the lines are detected, tag "Has Lines" is set and "lines_angle" feature
    is set to the value of best guess of lines angle in radians in range of
    [-pi/2; pi/2].
    """
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
            cv2.imwrite('../debug/%s_asrc.png' % name, shred)
            cv2.imwrite('../debug/%s_edges.png' % name, edges)
            cv2.imwrite('../debug/%s_mask.png' % name, mask)

        _, _, r_w, r_h = cv2.boundingRect(contour)

        # Line len should be at least 30% of shred's width, gap - 20%
        lines = cv2.HoughLinesP(edges, rho=10, theta=numpy.pi / 180 * 2,
                                threshold=30, maxLineGap=r_w * 0.2,
                                minLineLength=max([r_h, r_w]) * 0.3
                                )

        if lines is not None:
            lines = lines[0]
            tags.append(self.TAG_HAS_LINES_FEATURE)

            dominant_angle = _get_dominant_angle(lines)

            if DEBUG:
                dbg = cv2.cvtColor(edges, cv2.cv.CV_GRAY2BGRA)
                # Draw detected lines in green.
                for x1, y1, x2, y2 in lines:
                    cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0, 255), 1)

                approaches = [
                    ((0, 0, 255, 255), _get_dominant_angle(lines, MEAN)),
                    ((255, 0, 0, 255), _get_dominant_angle(lines, MEDIAN)),
                ]

                print [a[1] for a in approaches]

                # Draws lines originating from the middle of left border with
                # computed slopes: MEAN in red, MEDIAN in blue.
                for color, angle in approaches:
                    def y(x0, x):
                        return max(-2**15,
                                   min(2**15,
                                       int(x0 - math.tan(angle) * x)))

                    x0 = shred.shape[0]/2
                    x1 = 0
                    y1 = y(x0, x1)
                    x2 = shred.shape[1]
                    y2 = y(x0, x2)
                    cv2.line(dbg, (x1, y1), (x2, y2), color, 1)

                dbg = numpy.concatenate([shred, dbg], 1)
                cv2.imwrite('../debug/%s_houghlines.png' % name, dbg)
            params['lines_angle'] = dominant_angle

        return params, tags
