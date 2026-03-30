# ── Stage: production image ─────────────────────────────────────────────────
FROM python:3.11-slim

# System deps required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the actual requirements file (the root one just does -r backend/requirements.txt
# which breaks inside Docker — copy the real file directly instead)
COPY backend/requirements.txt ./requirements.txt

# Install CPU-only torch wheel first (much smaller than default CUDA bundle)
# then install the rest of the requirements
RUN pip install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Railway injects $PORT at runtime; Uvicorn binds to it
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use shell form so $PORT is expanded at container start
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
