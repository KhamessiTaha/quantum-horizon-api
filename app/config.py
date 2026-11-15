import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("QH_CACHE_DIR", "cache")).resolve())
    allowed_origins: list[str] = field(
        default_factory=lambda: _list(os.getenv("QH_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    )
    s3_bucket: str = os.getenv("QH_S3_BUCKET", "nasa-irsa-spherex")
    irsa_root: str = os.getenv("QH_IRSA_ROOT", "https://irsa.ipac.caltech.edu/")
    irsa_tap: str = os.getenv("QH_IRSA_TAP", "https://irsa.ipac.caltech.edu/TAP")
    irsa_sia: str = os.getenv("QH_IRSA_SIA", "https://irsa.ipac.caltech.edu/SIA")
    tap_timeout_s: float = float(os.getenv("QH_TAP_TIMEOUT_S", "20"))
    simbad_taps: list[str] = field(
        default_factory=lambda: _list(
            os.getenv(
                "QH_SIMBAD_TAPS",
                "https://simbad.cds.unistra.fr/simbad/sim-tap,https://simbad.cfa.harvard.edu/simbad/sim-tap",
            )
        )
    )
    read_workers: int = int(os.getenv("QH_READ_WORKERS", "8"))
    read_attempts: int = int(os.getenv("QH_READ_ATTEMPTS", "3"))
    s3_block_bytes: int = int(os.getenv("QH_S3_BLOCK_KB", "512")) * 1024
    max_frames: int = int(os.getenv("QH_MAX_FRAMES", "120"))
    max_frames_new: int = int(os.getenv("QH_MAX_FRAMES_NEW", os.getenv("QH_MAX_FRAMES", "120")))


settings = Settings()
