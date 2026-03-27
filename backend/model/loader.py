import os
import logging
import torch
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "unet_finetuned_v2.pth")
MODEL_PATH = os.getenv("UNET_MODEL_PATH", DEFAULT_MODEL_PATH)


def load_model():
    """
    Load the U-Net model with EfficientNet-B0 encoder.
    If the weights file is missing, the model runs with random
    (untrained) weights so the API pipeline can still be tested.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if os.path.isfile(MODEL_PATH):
        logger.info("Loading trained weights from %s", MODEL_PATH)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
    else:
        logger.warning(
            "Model weights not found at %s — running with UNTRAINED model. "
            "Predictions will be meaningless until real weights are provided.",
            MODEL_PATH,
        )

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
