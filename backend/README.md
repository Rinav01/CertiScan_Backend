# CertiScan Backend

Deepfake Document Detection API — U-Net segmentation model for detecting manipulation regions in document images. This README provides developer- and operator-facing details: architecture, installation, API schema, examples, testing, Docker usage, and troubleshooting.

## Quick summary

- Runtime: FastAPI app served by Uvicorn
- Model: U-Net (EfficientNet-B0 encoder) loaded via `segmentation_models_pytorch`
- Input: single image upload (multipart/form-data)
- Output: JSON with `prediction`, numeric `confidence`, `model_version`, and a downloadable mask served from `/outputs`

## Project files

- [backend/main.py](backend/main.py#L1) — application bootstrap, CORS and static mount for outputs.
- [backend/routes/predict.py](backend/routes/predict.py#L1) — `POST /predict` endpoint and request validation.
- [backend/model/loader.py](backend/model/loader.py#L1) — checkpoint loading, device selection, helper getters.
- [backend/utils/preprocess.py](backend/utils/preprocess.py#L1) — image decoding and tensor creation used by the model.
- [backend/utils/inference.py](backend/utils/inference.py#L1) — model forward pass and confidence calculation.
- [backend/config.py](backend/config.py#L1) — runtime configuration (MAX_UPLOAD_BYTES, outputs path, CORS origins).
- [backend/unet_finetuned_v2.pth](backend/unet_finetuned_v2.pth) — default checkpoint file (tracked here for convenience).

## Requirements

- Python 3.9+
- See `requirements.txt` for exact Python packages.

Install:

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
source .venv/bin/activate      # macOS / Linux
pip install -r backend/requirements.txt
```

## Configuration

From `backend/config.py` the notable configuration values are:

- `MAX_UPLOAD_BYTES` (default 10 MiB) — maximum accepted upload size.
- `OUTPUTS_DIR` — directory where mask PNGs are written and served.
- `DEFAULT_CORS_ORIGINS` — default origins allowed by CORS middleware.

Environment variables

- `UNET_MODEL_PATH` — override the checkpoint path. Example: `D:\models\my_unet.pth`.
- `ALLOW_UNTRAINED_MODEL` — set to `1` / `true` to allow server startup without compatible weights (development only).

Example (PowerShell):

```powershell
$env:UNET_MODEL_PATH = 'D:\models\unet_finetuned_v2.pth'
$env:ALLOW_UNTRAINED_MODEL = '1'
```

## Running the app (development)

From repository root:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at `http://127.0.0.1:8000/docs` for interactive OpenAPI UI.

## API details

### POST /predict

- Content-Type: `multipart/form-data`
- Field: `file` — the uploaded image (content-type must start with `image/`).
- Max size: `MAX_UPLOAD_BYTES` (10 MiB by default)

Server-side processing steps (see [backend/routes/predict.py](backend/routes/predict.py#L1)):

1. Validate upload content type and size.
2. Decode bytes to an RGB image and resize to 224×224 via `preprocess_image`.
3. Run the model forward pass (`run_inference`) to produce a binary mask (0/1).
4. Confidence = mean(mask) (see `calculate_confidence`).
5. Label = `Fake` if confidence > 0.1 else `Real` (threshold defined in route).
6. Save mask PNG to `outputs/` and return JSON response.

Response schema (Pydantic: [backend/schemas.py](backend/schemas.py#L1)):

```json
{
  "prediction": "Fake|Real",
  "confidence": 0.123,
  "threshold_used": 0.1,
  "model_version": "unet_finetuned_v2.pth",
  "mask_path": "outputs/<uuid>.png",
  "mask_url": "http://<host>/outputs/<uuid>.png"
}
```

Example `curl` request:

```bash
curl -F "file=@/path/to/document.jpg" http://127.0.0.1:8000/predict
```

Example Python client using `requests`:

```python
import requests

resp = requests.post('http://127.0.0.1:8000/predict', files={'file': open('doc.jpg','rb')})
print(resp.json())
```

Mask format

- The saved mask is a single-channel PNG where values are 0 or 255 (binary). The in-memory representation used for scoring is a float32 array with values in {0.0, 1.0}.

Thresholding and tuning

- The route currently uses a hardcoded threshold `PREDICTION_THRESHOLD = 0.1`.
- To change the threshold for a deployment, modify the value in `routes/predict.py` or add an environment/config-driven parameter.

## Model loading and behavior

- See [backend/model/loader.py](backend/model/loader.py#L1). The loader constructs the U-Net architecture and attempts to load a checkpoint from `MODEL_PATH` (resolved from `UNET_MODEL_PATH` or `unet_finetuned_v2.pth`).
- If the checkpoint is missing or incompatible, the server will fail to start unless `ALLOW_UNTRAINED_MODEL` is enabled.
- Device selection uses CUDA when `torch.cuda.is_available()`.

## Testing

- Unit tests are located under `backend/tests/`.
- `test_preprocess.py` verifies image decoding and tensor shapes (see [backend/tests/test_preprocess.py](backend/tests/test_preprocess.py#L1)).
- `test_routes.py` uses FastAPI's `TestClient` and mocks inference to verify the `/predict` payload and error handling (see [backend/tests/test_routes.py](backend/tests/test_routes.py#L1)).

Run tests:

```bash
pip install pytest
pytest -q
```

## Docker

A `Dockerfile` is included for containerized deployments. Build and run locally:

```bash
docker build -t certiscan-backend:latest .
docker run -p 8000:8000 \
  -e UNET_MODEL_PATH=/app/backend/unet_finetuned_v2.pth \
  -e CORS_ORIGINS="http://localhost:3000" \
  certiscan-backend:latest
```

## Railway deployment

This project is configured for easy deployment to [Railway](https://railway.app).

### Quick start

1. Push this repository to GitHub
2. Create a new Railway project and connect your GitHub repository
3. Railway will automatically detect `Procfile` and deploy the app
4. Set the following environment variables in Railway dashboard:
   - `UNET_MODEL_PATH`: path to your model checkpoint (default: `backend/unet_finetuned_v2.pth`)
   - `CORS_ORIGINS`: comma-separated list of frontend domain(s) (e.g., `https://myapp.com,https://www.myapp.com`)
   - `ALLOW_UNTRAINED_MODEL`: set to `0` for production (1 only for testing without a model)

### Prerequisites

- A trained U-Net checkpoint file (`unet_finetuned_v2.pth`) must be accessible or committed to the repo
- Python 3.10 (specified in `runtime.txt`)

### Configuration files

- `Procfile` — tells Railway how to run the app (Uvicorn with 0.0.0.0:$PORT)
- `runtime.txt` — specifies Python 3.10
- `.env.example` — template for environment variables
- `Dockerfile` — optional Docker build configuration
- `.dockerignore` — excludes unnecessary files from Docker build

### Monitoring and logs

- View logs in the Railway dashboard: **Project → Deployment → Logs**
- Configure alerts for deployment failures or high error rates

### Custom domain

Once deployed:
1. Go to Railway project settings
2. Add a custom domain (or use Railway's auto-generated domain)
3. Update your frontend `CORS_ORIGINS` in Railway to allow requests from your domain

## Deployment & scaling recommendations

- Prefer GPU hosts for higher throughput. Ensure `torch` + CUDA are installed in the environment.
- For production, run multiple Uvicorn worker processes behind a process manager (Gunicorn with uvicorn workers) or use an ASGI container platform.
- Protect `/predict` with authentication if exposing publicly; currently there is no auth.

## Logging & observability

- Application logging is configured in [backend/main.py](backend/main.py#L1). Integrate with a log aggregator or structured logging in production.
- Add metrics (Prometheus) around request latency and model inference time if needed.

## Common errors & troubleshooting

- Startup FileNotFoundError: checkpoint not found at `MODEL_PATH`. Fix `UNET_MODEL_PATH` or set `ALLOW_UNTRAINED_MODEL=1` (dev).
- 413 Payload Too Large: file exceeds `MAX_UPLOAD_BYTES` in `config.py`.
- 400 Bad Request: non-image uploads are rejected with message "Only image uploads are supported." (see tests asserting this behaviour).

## Contributing

- Create feature branches from `main`.
- Keep commits small and focused.
- Add unit tests for new logic and update `requirements.txt` when adding dependencies.

If you want, I can also:

- add a `Dockerfile` to the repo,
- add a small `Makefile` with common commands, or
- commit this README for you.

