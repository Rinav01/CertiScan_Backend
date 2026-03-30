import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    allow_credentials=DEFAULT_CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the outputs directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve saved masks as static files so they can be downloaded
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Include routes
app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unhandled exceptions.
    Starlette's CORSMiddleware skips header injection on unhandled exceptions,
    so we manually add CORS headers here so the browser sees the actual error
    instead of a misleading CORS block.
    """
    origin = request.headers.get("origin", "*")
    allowed = DEFAULT_CORS_ORIGINS == ["*"] or origin in DEFAULT_CORS_ORIGINS
    cors_origin = origin if allowed else ""

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": cors_origin or "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


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
