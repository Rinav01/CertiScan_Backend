# Vercel Deployment Guide — CertiScan Backend

## ⚠️ Important: Vercel Limitations

Vercel Serverless Functions have hard constraints that affect this ML backend:
- **60-second timeout** (maximum, even on Pro plan)
- **3GB memory limit** per function
- **Model loading time**: 15-30+ seconds for PyTorch models
- **No persistent storage** — `/outputs` directory won't work
- **Cold starts**: ~10-30s overhead (model reloaded each time)

### What works:
✅ Fast inference after model is warm  
✅ Low traffic API (fits within limits)  
✅ Demonstration/development

### What doesn't work:
❌ Production with high concurrency  
❌ Real-time predictions (cold start timeouts)  
❌ Storing output masks persistently  

## Quick Start

### 1. Push to GitHub
```bash
git add .
git commit -m "Add Vercel serverless deployment"
git push origin main
```

### 2. Create Vercel Project
- Go to [vercel.com](https://vercel.com)
- Connect GitHub repository
- Vercel auto-detects Python + `vercel.json` config

### 3. Set Environment Variables
In Vercel Dashboard → **Settings** → **Environment Variables**:

```
UNET_MODEL_PATH=backend/unet_finetuned_v2.pth
CORS_ORIGINS=https://yourfrontend.com
ALLOW_UNTRAINED_MODEL=0
```

### 4. Deploy
Push to GitHub → Vercel auto-deploys. Done!

## API Endpoints

### Health Check
```bash
curl https://<your-vercel-url>/api/health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2026-03-31T12:00:00",
  "service": "CertiScan Deepfake Detection API"
}
```

### Prediction
```bash
curl -F "file=@document.jpg" https://<your-vercel-url>/api/predict
```

Response:
```json
{
  "prediction": "Fake",
  "confidence": 0.85,
  "threshold_used": 0.1,
  "model_version": "unet_finetuned_v2.pth",
  "mask_data": "<hex-encoded-png>"
}
```

**Note:** Mask is returned as encoded data (no persistent `/outputs` storage on Vercel)

## Project Structure

```
api/
  ├── health.py      # Health check handler
  ├── predict.py     # Prediction handler
  ├── config.py      # Config for Vercel env
  └── __init__.py

backend/
  ├── main.py        # (Original FastAPI - not used on Vercel)
  ├── model/
  ├── utils/
  ├── routes/
  └── unet_finetuned_v2.pth

vercel.json           # Vercel configuration
.vercelignore         # Files to exclude from build
```

## Known Issues & Workarounds

### 1. Model Takes Too Long to Load
**Symptom:** `FUNCTION_TIMEOUT` error on first request  
**Cause:** PyTorch + model loading exceeds 60s  
**Workaround:**
- Upgrade to Vercel Pro for higher limits
- Pre-warm the function by calling `/api/health` periodically
- Use a background job to keep model loaded

### 2. Mask Storage
**Issue:** Can't save masks to `/outputs` (no persistent filesystem)  
**Solution:** Mask is returned as hex-encoded PNG data in response; decode on frontend

### 3. Cold Starts
**Issue:** First request after deployment/idle time is slow  
**Solution:** Use a cron service to periodically call `/api/health` to keep function warm

### 4. Memory Limits
**Issue:** Out of memory when loading model  
**Solution:**
- Upgrade to Vercel Pro (higher memory)
- Use model quantization to reduce size
- Split model inference into separate smaller functions

## Performance Tips

1. **Enable Vercel Pro** for higher limits (90s timeout, better specs)
2. **Use regional routing** — Vercel edge functions might be faster
3. **Monitor invocations** — Vercel Dashboard shows execution time
4. **Implement client-side retries** for timeout scenarios

## Monitoring

- **Vercel Dashboard** → **Analytics** — View request/error rates
- **Logs** → See function output and errors
- **Settings** → **Alerts** — Email on deployment failures

## Comparing Deployment Options

| Platform | Timeout | Storage | Cost | Best For |
|----------|---------|---------|------|----------|
| **Vercel** | 60s | Ephemeral | $20+/mo | Frontend + Simple APIs |
| **Railway** | Unlimited | Persistent | $5+/mo | Always-on Python services |
| **Render** | Unlimited | Persistent | $7+/mo | Background jobs + APIs |
| **Google Cloud Run** | 3600s | Ephemeral | Pay-per-use | Heavy workloads |

## Troubleshooting

### Build fails with "requirements.txt not found"
- Ensure `requirements.txt` exists at root and references `backend/requirements.txt`
- Check `.vercelignore` — might be excluding files

### API returns 500 error
- Check **Logs** in Vercel Dashboard
- Verify model file path in `UNET_MODEL_PATH` env var
- Ensure model file is committed to git or accessible

### Deployment slow
- Model takes time to package (3GB+ image)
- Consider Railway instead for persistent deployments

---

**Recommendation:** Use Vercel for the frontend, keep this backend on Railway for reliability. Vercel's 60s timeout is a hard blocker for ML inference at scale.
