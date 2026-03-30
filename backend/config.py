import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# CORS origins from environment variable (comma-separated)
# Default: localhost development origins
_CORS_ORIGINS_ENV = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
DEFAULT_CORS_ORIGINS = [origin.strip() for origin in _CORS_ORIGINS_ENV.split(",")]
