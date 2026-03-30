import logging
import os
from functools import lru_cache
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

try:
    from ..config import BASE_DIR
except ImportError:
    from config import BASE_DIR

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL_PATH = BASE_DIR / "unet_finetuned_v2.pth"
MODEL_PATH = Path(os.getenv("UNET_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
ALLOW_UNTRAINED_MODEL = os.getenv("ALLOW_UNTRAINED_MODEL", "").lower() in {
    "1",
    "true",
    "yes",
}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping.")

    return {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }


def load_model():
    """
    Load the U-Net model with EfficientNet-B0 encoder.
    Missing or incompatible weights fail startup by default so the API
    does not serve meaningless predictions.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if MODEL_PATH.is_file():
        logger.info("Loading trained weights from %s", MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = _extract_state_dict(checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint at {MODEL_PATH} is incompatible with the configured U-Net."
            ) from exc
    elif ALLOW_UNTRAINED_MODEL:
        logger.warning(
            "Model weights not found at %s - running with an untrained model.",
            MODEL_PATH,
        )
    else:
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Set UNET_MODEL_PATH to a "
            "valid checkpoint or ALLOW_UNTRAINED_MODEL=1 for development-only startup."
        )

    model.to(DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def get_model_status():
    return {
        "device": DEVICE,
        "model_path": str(MODEL_PATH),
        "weights_found": MODEL_PATH.is_file(),
        "allow_untrained_model": ALLOW_UNTRAINED_MODEL,
    }


def get_model_version():
    return MODEL_PATH.name
