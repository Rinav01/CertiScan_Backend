import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "backend"
# For Vercel: use /tmp for temporary files (ephemeral storage)
OUTPUTS_DIR = Path("/tmp/outputs") if os.getenv("VERCEL") else BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# CORS origins from environment variable (comma-separated)
_CORS_ORIGINS_ENV = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
DEFAULT_CORS_ORIGINS = [origin.strip() for origin in _CORS_ORIGINS_ENV.split(",")]

# For Vercel: allow longer timeout for model loading
VERCEL = os.getenv("VERCEL") == "1"
