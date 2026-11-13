"""Gaia DR3 sources in the field, propagated to each pass date.

Gaia positions are given at epoch J2016.0, and the stars this tool cares about move: Barnard's Star has travelled about
100 arcseconds since 2016, roughly 16 SPHEREx pixels. An overlay that ignored that would sit visibly off its stars, so
every source is propagated to the date of the pass it is drawn on.

A field holds hundreds of Gaia sources far below SPHEREx's depth, so the overlay keeps the brightest few and says so.
"""

import logging
import time

import astropy.units as u
import numpy as np
import pyvo
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time

from . import cache

log = logging.getLogger(__name__)

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
GAIA_EPOCH = Time("J2016.0")
DEFAULT_LIMIT_MAG = 17.0
MAX_SOURCES = 80
CACHE_DAYS = 30


class CatalogError(Exception):
    pass


def _query(ra: float, dec: float, radius_deg: float, limit_mag: float) -> list[dict]:
    adql = (
        f"SELECT TOP {MAX_SOURCES} source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_deg:.5f})) "
        f"AND phot_g_mean_mag < {limit_mag:.2f} "
        "ORDER BY phot_g_mean_mag ASC"
    )
    try:
        table = pyvo.dal.TAPService(GAIA_TAP).run_sync(adql, timeout=60).to_table()
    except Exception as error:  # ESA's archive, like any archive, has bad days
        raise CatalogError(f"Gaia archive unavailable: {type(error).__name__}: {str(error)[:160]}") from error

    rows = []
    for row in table:
        def value(name: str) -> float | None:
            raw = row[name]
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return None if number != number else number

        ra_value, dec_value = value("ra"), value("dec")
        if ra_value is None or dec_value is None:
            continue
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "ra": ra_value,
                "dec": dec_value,
                "pmra": value("pmra") or 0.0,
                "pmdec": value("pmdec") or 0.0,
                "parallax": value("parallax"),
                "g_mag": value("phot_g_mean_mag"),
            }
        )
    return rows


def sources(ra: float, dec: float, radius_deg: float, limit_mag: float = DEFAULT_LIMIT_MAG) -> tuple[list[dict], str]:
    """Gaia sources around a position, cached on disk. Returns the rows and where they came from."""
    name = cache.key("gaia", f"{ra:.4f}", f"{dec:.4f}", f"{radius_deg:.4f}", f"{limit_mag:.1f}")
    stored = cache.read_json("gaia", name)
    if stored and time.time() - stored.get("queried", 0) < CACHE_DAYS * 86400:
        return stored["sources"], "cache"
    try:
        rows = _query(ra, dec, radius_deg, limit_mag)
    except CatalogError:
        if stored:
            log.warning("Gaia unavailable; serving the stored catalogue for %.4f %.4f", ra, dec)
            return stored["sources"], "cache (Gaia unavailable)"
        raise
    cache.write_json("gaia", name, {"ra": ra, "dec": dec, "queried": time.time(), "sources": rows})
    return rows, "Gaia DR3"


def at_epoch(rows: list[dict], mjd: float) -> SkyCoord:
    """Propagate every source from Gaia's J2016.0 epoch to the date of an observation."""
    if not rows:
        return SkyCoord([], [], unit="deg")
    parallax = np.array([r["parallax"] if (r["parallax"] or 0) > 0.1 else 0.1 for r in rows])
    coords = SkyCoord(
        ra=[r["ra"] for r in rows] * u.deg,
        dec=[r["dec"] for r in rows] * u.deg,
        pm_ra_cosdec=[r["pmra"] for r in rows] * u.mas / u.yr,
        pm_dec=[r["pmdec"] for r in rows] * u.mas / u.yr,
        distance=Distance(parallax=parallax * u.mas, allow_negative=False),
        radial_velocity=np.zeros(len(rows)) * u.km / u.s,
        obstime=GAIA_EPOCH,
        frame="icrs",
    )
    moved = coords.apply_space_motion(new_obstime=Time(mjd, format="mjd"))
    return SkyCoord(moved.ra, moved.dec, frame="icrs")


def moves_more_than(row: dict, arcsec_per_year: float) -> bool:
    return float(np.hypot(row["pmra"], row["pmdec"])) / 1000.0 >= arcsec_per_year
