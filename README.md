# CertiScan Backend

> Deepfake Document Detection API — AI-powered detection of manipulated document images using U-Net segmentation.

## Overview

CertiScan Backend is a FastAPI-based REST API for detecting document forgery and deepfake manipulation. Upload an image and receive a confidence score and visual mask showing potentially manipulated regions.

- **Model**: U-Net with EfficientNet-B0 encoder
- **Framework**: FastAPI + Uvicorn
- **Deployment**: Ready for Railway, Docker, or traditional hosting

## Quick start (local development)

```bash
# Clone and navigate to project
cd backend

# Create and activate virtual environment
python -m venv .venv
. .venv/Scripts/Activate.ps1    # Windows PowerShell
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

Server will be at `http://127.0.0.1:8000`

View interactive API docs: `http://127.0.0.1:8000/docs`

## Quick start (Docker)

```bash
docker build -t certiscan:latest .
docker run -p 8000:8000 certiscan:latest
```

## Deploying to Railway

1. Push to GitHub
2. Connect repo to [Railway](https://railway.app)
3. Set environment variables (see `.env.example`)
4. Deploy — Railway handles the rest via `Procfile`

**See [backend/README.md](backend/README.md) for detailed Railway setup instructions.**

## API usage

### Check health
```bash
curl http://localhost:8000/health
```

### Predict document authenticity
```bash
curl -F "file=@document.jpg" http://localhost:8000/predict
```

Response:
```json
{
  "prediction": "Real",
  "confidence": 0.95,
  "threshold_used": 0.1,
  "model_version": "unet_finetuned_v2.pth",
  "mask_path": "outputs/abc123.png",
  "mask_url": "http://localhost:8000/outputs/abc123.png"
}
```

## Environment variables

See `.env.example` for all available options:

- `UNET_MODEL_PATH` — path to model checkpoint
- `CORS_ORIGINS` — comma-separated list of allowed CORS origins
- `ALLOW_UNTRAINED_MODEL` — allow running without trained weights (dev only)

## Project structure

```
.
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── routes/predict.py       # Prediction endpoint
│   ├── model/loader.py         # Model loading
│   ├── utils/
│   │   ├── preprocess.py       # Image preprocessing
│   │   └── inference.py        # Model inference
│   ├── unet_finetuned_v2.pth   # Model weights
│   └── README.md               # Detailed backend docs
├── Procfile                    # Railway config
├── runtime.txt                 # Python version
├── Dockerfile                  # Container config
├── requirements.txt            # Python dependencies
└── .env.example               # Environment template
```

## Testing

```bash
pip install pytest
pytest -q
```

## Troubleshooting

See [backend/README.md](backend/README.md#troubleshooting) for detailed troubleshooting guide.

Common issues:
- **Model not found**: Set `UNET_MODEL_PATH` or use `ALLOW_UNTRAINED_MODEL=1` (dev)
- **CORS errors**: Update `CORS_ORIGINS` environment variable
- **Port in use**: Change port in development or check for existing process

## Contributing

- Create feature branches from `main`
- Keep commits focused and well-documented
- Add unit tests for new features
- Update `requirements.txt` when adding dependencies

## License

[Add license info here]

---

**For detailed backend documentation, see [backend/README.md](backend/README.md)**
