"""Vercel entry point.

Vercel functions can only write to /tmp, so the committed cache is copied there on a cold start and astropy is given
/tmp as its home. Locally and on Fly this file is unused; run `uvicorn app.main:app` as before.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE = Path("/tmp/qh-cache")

os.environ["HOME"] = "/tmp"
os.environ.setdefault("QH_CACHE_DIR", str(RUNTIME_CACHE))
if not RUNTIME_CACHE.exists():
    shutil.copytree(ROOT / "cache", RUNTIME_CACHE, dirs_exist_ok=True)

sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from app.main import app as fastapi_app  # noqa: E402


async def app(scope, receive, send):
    if scope["type"] == "http":
        raw = scope.get("raw_path") or b""
        raw_path = raw.decode("latin-1").split("?", 1)[0]
        if b"qh_debug=1" in scope.get("query_string", b""):
            body = json.dumps(
                {
                    "path": scope.get("path"),
                    "raw_path": raw_path,
                    "root_path": scope.get("root_path"),
                    "headers": {k.decode(): v.decode() for k, v in scope.get("headers", []) if k.startswith(b"x-")},
                }
            ).encode()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        if raw_path.startswith("/api/") and scope.get("path") != raw_path:
            scope = {**scope, "path": raw_path, "root_path": ""}
    await fastapi_app(scope, receive, send)
