import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings


def key(*parts: Any) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def path(kind: str, name: str) -> Path:
    folder = settings.cache_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def read_json(kind: str, name: str) -> Any | None:
    target = path(kind, f"{name}.json")
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(kind: str, name: str, value: Any) -> None:
    _atomic_write(path(kind, f"{name}.json"), json.dumps(value, indent=1).encode("utf-8"))


def read_arrays(kind: str, name: str) -> dict[str, np.ndarray] | None:
    target = path(kind, f"{name}.npz")
    if not target.exists():
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except (OSError, ValueError):
        return None


def write_arrays(kind: str, name: str, **arrays: np.ndarray) -> None:
    target = path(kind, f"{name}.npz")
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_bytes(kind: str, name: str, data: bytes) -> Path:
    target = path(kind, name)
    _atomic_write(target, data)
    return target
