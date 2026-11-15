"""Vercel entry point.

Vercel functions can only write to /tmp, so the committed cache is copied there on a cold start and astropy is given
/tmp as its home. A Hobby function has one shared CPU and 300 s per request, so new positions read at most 85 frames (the largest featured view uses 84),
spread evenly in time, and a hanging IRSA TAP is abandoned after 10 s for its SIA service. Locally and on Fly this file is unused; run `uvicorn app.main:app` as before.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE = Path("/tmp/qh-cache")

os.environ["HOME"] = "/tmp"
os.environ.setdefault("QH_CACHE_DIR", str(RUNTIME_CACHE))
os.environ.setdefault("QH_MAX_FRAMES", "85")
os.environ.setdefault("QH_TAP_TIMEOUT_S", "10")
if not RUNTIME_CACHE.exists():
    shutil.copytree(ROOT / "cache", RUNTIME_CACHE, dirs_exist_ok=True)

sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402,F401
