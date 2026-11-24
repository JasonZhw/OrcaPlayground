"""Check raw-camera geometry using the actual installed official image tool."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_preprocessing import prepare_camera

import numpy as np
from image_preprocessing import resize_with_pad


class PreprocessingTests(unittest.TestCase):
    def test_padding_geometry_and_rgb(self):
        image = np.empty((480, 640, 3), dtype=np.uint8)
        image[:] = [17, 91, 203]
        result = prepare_camera(image)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(result[:28].any())
        self.assertFalse(result[196:].any())
        np.testing.assert_array_equal(result[28:196], np.broadcast_to([17, 91, 203], (168, 224, 3)))

    def test_square_remains_square(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[160:320, 240:400, 0] = 255
        result = prepare_camera(image)
        rows, columns = np.nonzero(result[:, :, 0] > 127)
        self.assertEqual(int(rows.max() - rows.min()), int(columns.max() - columns.min()))
        self.assertAlmostEqual(float(rows.mean()), 111.5)
        self.assertAlmostEqual(float(columns.mean()), 111.5)

    def test_camera_order_and_official_pixels(self):
        pair = np.empty((480, 1280, 3), dtype=np.uint8)
        pair[:, :640] = [255, 0, 0]
        pair[:, 640:] = [0, 0, 255]
        head = prepare_camera(pair[:, :640])
        wrist = prepare_camera(pair[:, 640:])
        np.testing.assert_array_equal(head[112, 112], [255, 0, 0])
        np.testing.assert_array_equal(wrist[112, 112], [0, 0, 255])
        np.testing.assert_array_equal(head, resize_with_pad(pair[:, :640], 224, 224))

    def test_rejects_wrong_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            prepare_camera(np.zeros((480, 640, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            prepare_camera(np.zeros((224, 224, 3), dtype=np.uint8))
