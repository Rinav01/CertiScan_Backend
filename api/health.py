import json
import os
from datetime import datetime

# Health check endpoint for Vercel
def handler(request):
    """Simple health check - fast response"""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "CertiScan Deepfake Detection API"
        })
    }
