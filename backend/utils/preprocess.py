import cv2
import numpy as np
import torch


def preprocess_image(file_bytes: bytes, size: int = 224):
    """
    Decode raw image bytes and prepare a normalised tensor for the model.

    Returns
    -------
    tensor : torch.Tensor  — shape (1, 3, H, W), float32, values in [0, 1]
    image  : np.ndarray     — the resized RGB image (H, W, 3), values in [0, 1]
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded file as an image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size)).astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, image
