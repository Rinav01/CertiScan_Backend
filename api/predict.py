import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

# Import backend modules
import sys
sys.path.insert(0, '/var/task')

from backend.utils.preprocess import preprocess_image
from backend.utils.inference import run_inference, calculate_confidence
from backend.model.loader import get_model

# Store model in memory for reuse across invocations
_model_cache = {}

def get_cached_model():
    """Cache model instance across Vercel invocations"""
    if "model" not in _model_cache:
        try:
            _model_cache["model"] = get_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return _model_cache["model"]

def handler(request):
    """
    Vercel serverless function for image prediction.
    Handle as: POST /api/predict with multipart/form-data file upload
    """
    
    if request.method != "POST":
        return {
            "statusCode": 405,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Method not allowed"})
        }
    
    try:
        # Parse multipart form data
        files = request.files
        if "file" not in files:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "No file provided"})
            }
        
        file = files["file"]
        
        # Validate content type
        if not file.content_type or not file.content_type.startswith("image/"):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Only image uploads are supported"})
            }
        
        # Read file bytes
        file_bytes = file.read()
        
        # Check MAX_UPLOAD_BYTES (10MB)
        MAX_UPLOAD_BYTES = 10 * 1024 * 1024
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return {
                "statusCode": 413,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": f"File too large. Max {MAX_UPLOAD_BYTES} bytes."
                })
            }
        
        # Preprocess image
        try:
            image_tensor, _ = preprocess_image(file_bytes)
        except ValueError as e:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(e)})
            }
        
        # Run inference
        model = get_cached_model()
        mask = run_inference(image_tensor)
        confidence = calculate_confidence(mask)
        
        # Determine label
        PREDICTION_THRESHOLD = 0.1
        label = "Fake" if confidence > PREDICTION_THRESHOLD else "Real"
        
        # Note: In Vercel, we can't persist files in /outputs
        # Return mask as base64 or data URI instead
        mask_uint8 = (mask * 255).astype(np.uint8)
        _, encoded = cv2.imencode(".png", mask_uint8)
        mask_base64 = encoded.tobytes().hex()  # Simple hex encoding
        
        response = {
            "prediction": label,
            "confidence": round(float(confidence), 3),
            "threshold_used": PREDICTION_THRESHOLD,
            "model_version": "unet_finetuned_v2.pth",
            "mask_data": mask_base64,  # Base64/hex encoded PNG
            "note": "Mask is returned as encoded data (Vercel has no persistent /outputs)"
        }
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response)
        }
    
    except RuntimeError as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Model error: {str(e)}"})
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Server error: {str(e)}"})
        }
