"""Band overlay_wannier / overlay_bands schema contract."""
import unittest

from tensorspec.web.server.schemas import BandRequest, BandResult


class TestBandOverlaySchema(unittest.TestCase):
    def test_overlay_wannier_default_false(self):
        self.assertFalse(BandRequest().overlay_wannier)

    def test_overlay_bands_optional(self):
        # Minimal BandResult-like construction may need many fields —
        # assert field exists on model:
        self.assertIn("overlay_bands", BandResult.model_fields)
        self.assertIn("overlay_wannier", BandRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
