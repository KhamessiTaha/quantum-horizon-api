"""Find every SPHEREx spectral image whose footprint covers a sky position.

IRSA TAP sees new weekly releases first. Every list found is stored, so a position seen once can be served again
from the S3 mirror alone if IRSA is down (observed on 4 Nov 2026). When TAP hangs (observed on 15 Nov 2026, while the
rest of IRSA answered), the same images are found through IRSA's SIA service instead, which lists QR3 and DR1 but not
the older QR2.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pyvo

from . import cache
from .config import settings
from .http import TimeoutSession
from .models import ImageRecord

log = logging.getLogger(__name__)

RELEASE = re.compile(r"/((?:qr|dr)\d+)/", re.IGNORECASE)
DETECTOR = re.compile(r"SPHEREx-(D[1-6])")
SIA_COLLECTIONS = ("spherex_qr3", "spherex_dr1")
TAP_BACKOFF_S = 300.0
SIA_TIMEOUT_S = 60
SIA_BUDGET_S = 150
SIA_PAUSE_S = 2

_tap_down_until = 0.0


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
    service = pyvo.dal.TAPService(settings.irsa_tap, session=TimeoutSession(settings.tap_timeout_s))
    try:
        table = service.run_sync(adql).to_table()
    except Exception as error:
        raise DiscoveryError(f"IRSA TAP unavailable: {type(error).__name__}: {str(error)[:160]}") from error

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


def _sia_collection(ra: float, dec: float, collection: str) -> list[ImageRecord]:
    """IRSA's SIA fails in bursts of instant 502s between slow successes, so keep asking until the time budget is spent."""
    deadline = time.monotonic() + SIA_BUDGET_S
    attempt = 0
    while True:
        attempt += 1
        try:
            service = pyvo.dal.SIA2Service(settings.irsa_sia, session=TimeoutSession(SIA_TIMEOUT_S), check_baseurl=False)
            table = service.search(pos=(ra, dec, 0.0001), collection=collection).to_table()
            break
        except Exception as error:
            if time.monotonic() + SIA_PAUSE_S >= deadline:
                raise DiscoveryError(
                    f"IRSA SIA unavailable after {attempt} tries: {type(error).__name__}: {str(error)[:160]}"
                ) from error
            log.warning("IRSA SIA %s failed (%s); retrying", collection, type(error).__name__)
            time.sleep(SIA_PAUSE_S)
    records = []
    for row in table:
        if str(row["dataproduct_subtype"]) != "science":
            continue
        uri = str(row["access_url"])
        detector = DETECTOR.search(str(row["energy_bandpassname"]))
        if not uri.lower().endswith(".fits") or not detector:
            continue
        records.append(
            ImageRecord(
                plane_id=str(row["obs_publisher_did"]),
                mjd=float(row["t_min"]),
                detector=detector.group(1),
                release=release_of(uri),
                uri=uri,
            )
        )
    return records


def _query_sia(ra: float, dec: float) -> list[ImageRecord]:
    """Both collections at once: SIA answers in about 20 s per query, so asking one after the other doubles the wait."""
    with ThreadPoolExecutor(max_workers=len(SIA_COLLECTIONS)) as pool:
        parts = list(pool.map(lambda c: _sia_collection(ra, dec, c), SIA_COLLECTIONS))
    return sorted((r for part in parts for r in part), key=lambda r: r.mjd)


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
    global _tap_down_until
    records, via = None, "IRSA TAP"
    if time.time() >= _tap_down_until:
        try:
            records = _query_tap(ra, dec)
        except DiscoveryError as error:
            _tap_down_until = time.time() + TAP_BACKOFF_S
            log.warning("%s; using SIA for the next %.0f s", error, TAP_BACKOFF_S)
    if records is None:
        if stored:
            log.warning("IRSA TAP down; serving the stored image list for %.4f %.4f", ra, dec)
            return _stored(stored["images"]), "stored (IRSA unavailable)"
        records, via = _query_sia(ra, dec), "IRSA SIA"
    cache.write_json("index", name, {"ra": ra, "dec": dec, "queried": time.time(), "images": [r.model_dump() for r in records]})
    return records, via
