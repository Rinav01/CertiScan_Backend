"""
Vercel ASGI wrapper for local testing.
Run with: vercel dev
"""

from backend.main import app

# For Vercel serverless - export ASGI app
asgi_app = app
