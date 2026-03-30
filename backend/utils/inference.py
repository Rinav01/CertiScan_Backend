import numpy as np
import torch

try:
    from ..model.loader import DEVICE, get_model
except ImportError:
    from model.loader import DEVICE, get_model


def run_inference(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Run the U-Net forward pass and return a binary mask.

    Parameters
    ----------
    image_tensor : torch.Tensor - shape (1, 3, H, W)

    Returns
    -------
    mask : np.ndarray - shape (H, W), binary float32 values {0.0, 1.0}
    """
    image_tensor = image_tensor.to(DEVICE)
    model = get_model()

    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).float().cpu().numpy()[0][0]

    return mask


def calculate_confidence(mask: np.ndarray) -> float:
    """Return the mean of the binary mask as a confidence score (0-1)."""
    return float(mask.mean())
