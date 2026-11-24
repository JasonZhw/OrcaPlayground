"""Shared raw-camera preprocessing for conversion and portable inference."""

import importlib.util
from pathlib import Path


source = Path(__file__).resolve().parents[1] / "third_party/openpi/image_tools.py"
spec = importlib.util.spec_from_file_location("v9_openpi_image_tools", source)
image_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_tools)
resize_with_pad = image_tools.resize_with_pad


def prepare_camera(image):
    import numpy as np

    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError("Expected uint8[480,640,3] RGB camera observation")
    return image_tools.resize_with_pad(image, 224, 224)
