import unittest

import cv2
import numpy as np

from backend.utils.preprocess import preprocess_image


class PreprocessImageTests(unittest.TestCase):
    def test_preprocess_returns_expected_tensor_shape(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        tensor, resized = preprocess_image(encoded.tobytes())

        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertEqual(resized.shape, (224, 224, 3))

    def test_preprocess_rejects_invalid_image_bytes(self):
        with self.assertRaises(ValueError):
            preprocess_image(b"not-an-image")


if __name__ == "__main__":
    unittest.main()
