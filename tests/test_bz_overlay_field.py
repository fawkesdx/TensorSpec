"""BZRequest no longer carries unused overlay_crystal."""
import unittest

from tensorspec.web.server.schemas import BZRequest


class TestBZRequestNoOverlayField(unittest.TestCase):
    def test_model_has_no_overlay_crystal(self):
        self.assertNotIn("overlay_crystal", BZRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
