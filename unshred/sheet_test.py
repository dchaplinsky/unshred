import hashlib
import os.path
import shutil
import tempfile
import unittest

import numpy

from split import SheetIO


_TEST_IMAGE = os.path.join(os.path.dirname(__file__), '..', 'src', '10.jpg')


class SheetTest(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        self.sheet_io = SheetIO(_TEST_IMAGE, 'test sheet',
                                [], self.out_dir, 'png')
        self.sheet = self.sheet_io.sheet

    def test_guess_dpi(self):
        dpi = self.sheet._guess_dpi()
        self.assertEqual(dpi, (300, 300))

        self.sheet.orig_img = numpy.zeros((100, 200))
        self.assertRaises(ValueError, self.sheet._guess_dpi)

    def test_detect_scanner_background(self):
        expected_bg_mask_hash = 'd97d995764654e496183a125e7400008'
        bg_mask = self.sheet._detect_scanner_background()
        self.assertEqual(
                hashlib.md5(bg_mask.tostring()).hexdigest(),
                expected_bg_mask_hash)

    def test_foreground_mask(self):
        expected_fg_mask_hash = 'd42019bbfedd754fe63b0cc0093be887'
        fg_mask = self.sheet._find_foreground_mask()
        self.assertEqual(
                hashlib.md5(fg_mask.tostring()).hexdigest(),
                expected_fg_mask_hash)


    def test_get_shreds(self):
        shreds = self.sheet.get_shreds([], 'test sheet')
        self.assertEqual(len(shreds), 209)

        shred = shreds[0]
        self.assertEqual(len(shred.contour), 190)
        self.assertDictEqual(shred.features, {
            'pos_height': 54,
            'pos_x': 1126,
            'pos_y': 3411,
            'angle': 89.84802217781544,
            'pos_width': 462,
        })

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
