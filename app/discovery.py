"""Find every SPHEREx spectral image whose footprint covers a sky position.

IRSA TAP sees new weekly releases first. Every list found is stored, so a position seen once can be served again
from the S3 mirror alone if IRSA is down (observed on 4 Nov 2026).
"""

import logging
import re
import time

import pyvo

from . import cache
from .config import settings
from .models import ImageRecord

log = logging.getLogger(__name__)

RELEASE = re.compile(r"/((?:qr|dr)\d+)/", re.IGNORECASE)
DETECTOR = re.compile(r"SPHEREx-(D[1-6])")


class DiscoveryError(Exception):
    pass


def _index_name(ra: float, dec: float) -> str:
    return cache.key("index", f"{ra:.4f}", f"{dec:.4f}")


def _query_tap(ra: float, dec: float) -> list[ImageRecord]:
    adql = (
        "SELECT p.planeid, p.time_bounds_lower, p.energy_bandpassname, a.uri "
        "FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
        f"WHERE CONTAINS(POINT('ICRS', {ra:.7f}, {dec:.7f}), p.poly) = 1"
    )
    service = pyvo.dal.TAPService(settings.irsa_tap)
    last: Exception | None = None
    for attempt in range(3):
        try:
            table = service.run_sync(adql, timeout=90).to_table()
            break
        except Exception as error:
            last = error
            time.sleep(2 * (attempt + 1))
    else:
        raise DiscoveryError(f"IRSA TAP unavailable: {type(last).__name__}: {str(last)[:160]}")

    records = []
    for row in table:
        uri = str(row["uri"])
        if not uri.lower().endswith(".fits"):
            continue
        detector = DETECTOR.search(str(row["energy_bandpassname"]))
        if not detector:
            continue
        records.append(
            ImageRecord(
                plane_id=str(row["planeid"]),
                mjd=float(row["time_bounds_lower"]),
                detector=detector.group(1),
                release=release_of(uri),
                uri=uri,
            )
        )
    return sorted(records, key=lambda r: r.mjd)


def release_of(uri: str) -> str:
    """The release a file belongs to, from its path: qr2, qr3 for quick releases, dr1 for the first data release."""
    found = RELEASE.search(uri)
    return found.group(1).lower() if found else "unknown"


def _stored(images: list[dict]) -> list[ImageRecord]:
    return [ImageRecord(**{**r, "release": release_of(r["uri"])}) for r in images]


def find_images(ra: float, dec: float, refresh: bool = False) -> tuple[list[ImageRecord], str]:
    name = _index_name(ra, dec)
    stored = cache.read_json("index", name)
    if stored and not refresh:
        return _stored(stored["images"]), "stored"
    try:
        records = _query_tap(ra, dec)
    except DiscoveryError:
        if stored:
            log.warning("IRSA TAP down; serving the stored image list for %.4f %.4f", ra, dec)
            return _stored(stored["images"]), "stored (IRSA unavailable)"
        raise
    cache.write_json("index", name, {"ra": ra, "dec": dec, "queried": time.time(), "images": [r.model_dump() for r in records]})
    return records, "IRSA TAP"
