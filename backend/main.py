import os
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.predict import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Document Detection API",
    description="Upload a document image to detect potential deepfake manipulation.",
    version="0.1.0",
)

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Serve saved masks as static files so they can be downloaded
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Include routes
app.include_router(router)


@app.get("/")
def home():
    return {"message": "Deepfake Document Detection API"}
