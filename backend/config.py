import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# CORS origins from environment variable (comma-separated).
# On Railway, set CORS_ORIGINS to your Vercel frontend URL, e.g.:
#   CORS_ORIGINS=https://your-app.vercel.app
# Defaults to "*" (allow all) when not set, so Railway works without config.
_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "*")

if _CORS_ORIGINS_ENV.strip() == "*":
    DEFAULT_CORS_ORIGINS = ["*"]
else:
    DEFAULT_CORS_ORIGINS = [origin.strip() for origin in _CORS_ORIGINS_ENV.split(",")]
