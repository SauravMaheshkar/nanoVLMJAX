from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor


def process_image(processor: AutoProcessor, image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="np")
    return inputs["pixel_values"]


def process_sample_to_np(
    sample: dict[str, Any], dtype: np.dtype = np.float32
) -> dict[str, Any]:
    result = {}
    for key, value in sample.items():
        if value is None:
            result[key] = None
        elif isinstance(value, (str, int, float)):
            result[key] = np.asarray(value, dtype=dtype)
        elif isinstance(value, list):
            result[key] = np.asarray(value, dtype=dtype)
        elif hasattr(value, "numpy"):
            result[key] = value.numpy()
        else:
            result[key] = np.asarray(value, dtype=dtype)
    return result
