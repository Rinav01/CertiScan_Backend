import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from .config import DEFAULT_CORS_ORIGINS, OUTPUTS_DIR
    from .model.loader import get_model, get_model_status
    from .schemas import HealthResponse
    from .routes.predict import router
except ImportError:
    from config import DEFAULT_CORS_ORIGINS, OUTPUTS_DIR
    from model.loader import get_model, get_model_status
    from schemas import HealthResponse
    from routes.predict import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Document Detection API",
    description="Upload a document image to detect potential deepfake manipulation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the outputs directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve saved masks as static files so they can be downloaded
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Include routes
app.include_router(router)


@app.on_event("startup")
def warm_up_model():
    """Fail fast during startup if model loading is misconfigured."""
    get_model()


@app.get("/")
def home():
    return {"message": "Deepfake Document Detection API"}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", **get_model_status()}
