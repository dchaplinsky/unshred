import cv2

from .base import AbstractShredFeature


class GeometryFeatures(AbstractShredFeature):
    def get_info(self, shred, contour, name):
        area = cv2.contourArea(contour)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        solidity = float(area) / hull_area

        # Also top and bottom points on the contour
        topmost = map(int, contour[contour[:, :, 1].argmin()][0])
        bottommost = map(int, contour[contour[:, :, 1].argmax()][0])
        _, _, r_w, r_h = cv2.boundingRect(contour)
        tags = []

        if solidity < 0.75:
            tags.append("Suspicious shape")

        width_mm = float(r_w) / self.sheet.res_x * 25.4
        height_mm = float(r_h) / self.sheet.res_y * 25.4
        ratio = shred.shape[0] / float(shred.shape[1])

        if width_mm > 6:
            tags.append("Suspicious width")

        return {
            "area": area,
            "ratio": ratio,
            "solidity": solidity,
            "topmost": topmost,
            "bottommost": bottommost,
            "width_mm": width_mm,
            "height_mm": height_mm
        }, tags


# # Let's also detect 25 of most outstanding corners of the contour.
# corners = cv2.goodFeaturesToTrack(mask, 25, 0.01, 10)
# corners = np.int0(corners) if corners is not None else []

# # edges = cv2.Canny(img_roi[:, :, 2], 100, 200)
# # save_image("pieces/%s_edges" % name, edges)

# # Draw contours for debug purposes
# mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGRA)
# mask[:, :, 3] = mask[:, :, 0]
# cv2.drawContours(mask, contours, -1, (255, 0, 255, 255), 2)

# # And put corners on top
# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(mask, (x, y), 3, (0, 255, 0, 255), -1)

# cnt = contours[0]

# # Let's find features that will help us to determine top side of the piece
# # if mask.shape[0] / float(mask.shape[1]) >= 2.7:
# hull = cv2.convexHull(cnt, returnPoints=False)
# defects = cv2.convexityDefects(cnt, hull)

# # Check for convex defects to see if we can find a trace of a
# # shredder cut which is usually on top of the piece.

# if defects is not None:
#     has_defects = False
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         far = tuple(cnt[f][0])

#         # if convex defect is big enough
#         if d / 256. > 50:
#             has_defects = True
#             cv2.circle(mask, far, 5, [0, 0, 255, 255], -1)

#             # # And lays at the top or bottom of the piece
#             # y_dist = min(abs(0 - far[1]), abs(mask.shape[0] - far[1]))
#             # if float(y_dist) / mask.shape[0] < 0.1:
#             #     # and more or less is in the center
#             #     if abs(far[0] - mask.shape[1] / 2.) / mask.shape[1] < 0.25:
#             #         cv2.circle(mask, far, 5, [0, 0, 255, 255], -1)

#     # if has_defects:
#     #     cv2.imwrite("debug/def_20_%s_%s.png" % (self.sheet_name, name), mask)