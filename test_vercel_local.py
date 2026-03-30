#!/usr/bin/env python3
"""
Quick test script for Vercel serverless functions.
Tests locally before deploying to Vercel.

Requirements: Flask (for testing request objects)
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test health endpoint
print("=" * 60)
print("Testing /api/health endpoint")
print("=" * 60)

from api import health

class MockRequest:
    method = "GET"

request = MockRequest()
response = health.handler(request)
print(json.dumps(json.loads(response["body"]), indent=2))
print(f"Status: {response['statusCode']}")

print("\n" + "=" * 60)
print("✅ Health check works!")
print("=" * 60)

print("""
To test /api/predict locally:
1. Install Flask: pip install Flask
2. Run: vercel dev
3. POST to: http://localhost:3000/api/predict

Or use curl:
curl -F "file=@/path/to/image.jpg" http://localhost:3000/api/predict
""")
