import uuid

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from utils.preprocess import preprocess_image
from utils.inference import run_inference, calculate_confidence

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload, run deepfake detection, and return
    the prediction label, confidence score, and path to the
    generated mask image.
    """
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image_tensor, _original = preprocess_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    mask = run_inference(image_tensor)
    confidence = calculate_confidence(mask)

    label = "Fake" if confidence > 0.1 else "Real"

    # Save the mask as a PNG
    mask_filename = f"{uuid.uuid4()}.png"
    mask_path = f"outputs/{mask_filename}"
    cv2.imwrite(mask_path, (mask * 255).astype("uint8"))

    return {
        "prediction": label,
        "confidence": round(confidence, 3),
        "mask_path": mask_path,
    }
