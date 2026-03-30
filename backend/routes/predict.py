from pathlib import Path
import uuid

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

try:
    from ..config import MAX_UPLOAD_BYTES, OUTPUTS_DIR
    from ..model.loader import get_model_version
    from ..schemas import PredictResponse
    from ..utils.inference import calculate_confidence, run_inference
    from ..utils.preprocess import preprocess_image
except ImportError:
    from config import MAX_UPLOAD_BYTES, OUTPUTS_DIR
    from model.loader import get_model_version
    from schemas import PredictResponse
    from utils.inference import calculate_confidence, run_inference
    from utils.preprocess import preprocess_image

router = APIRouter()
PREDICTION_THRESHOLD = 0.1


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accept an image upload, run deepfake detection, and return
    the prediction label, confidence score, and path to the
    generated mask image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file is too large. Max size is {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        image_tensor, _original = preprocess_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    mask = run_inference(image_tensor)
    confidence = calculate_confidence(mask)

    label = "Fake" if confidence > PREDICTION_THRESHOLD else "Real"

    OUTPUTS_DIR.mkdir(exist_ok=True)
    mask_filename = f"{uuid.uuid4()}.png"
    output_file = OUTPUTS_DIR / mask_filename

    if not cv2.imwrite(str(output_file), (mask * 255).astype("uint8")):
        raise HTTPException(status_code=500, detail="Failed to write prediction mask.")

    mask_path = str(Path("outputs") / mask_filename).replace("\\", "/")
    mask_url = str(request.base_url).rstrip("/") + "/" + mask_path

    return {
        "prediction": label,
        "confidence": round(confidence, 3),
        "threshold_used": PREDICTION_THRESHOLD,
        "model_version": get_model_version(),
        "mask_path": mask_path,
        "mask_url": mask_url,
    }
