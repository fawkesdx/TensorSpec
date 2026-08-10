import unittest
import numpy as np
from tensorspec.core.peem_roi import roi_to_mask


class TestPeemRoi(unittest.TestCase):
    def test_rect_inclusive(self):
        m = roi_to_mask(5, 5, {"kind": "rect", "x0": 1, "y0": 1, "x1": 2, "y1": 2})
        self.assertEqual(int(m.sum()), 4)
        self.assertTrue(m[1, 1] and m[2, 2])
        self.assertFalse(m[0, 0])

    def test_rect_normalizes_order(self):
        m = roi_to_mask(5, 5, {"kind": "rect", "x0": 3, "y0": 3, "x1": 1, "y1": 1})
        self.assertEqual(int(m.sum()), 9)

    def test_ellipse_center(self):
        m = roi_to_mask(7, 7, {"kind": "ellipse", "cx": 3, "cy": 3, "rx": 2, "ry": 1})
        self.assertTrue(m[3, 3])
        self.assertFalse(m[0, 0])
        self.assertGreater(int(m.sum()), 0)

    def test_polygon_triangle(self):
        m = roi_to_mask(
            6,
            6,
            {"kind": "polygon", "points": [[1, 1], [4, 1], [1, 4]]},
        )
        self.assertTrue(m[2, 1])
        self.assertGreater(int(m.sum()), 3)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            roi_to_mask(4, 4, {"kind": "rect", "x0": 10, "y0": 10, "x1": 12, "y1": 12})
