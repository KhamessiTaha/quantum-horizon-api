"""Name and coordinate resolution, and proper-motion propagation.

Catalog positions are referenced to J2000.0. Nearby stars move by arcminutes since then (Barnard's Star: 270" by 2026),
so every comparison against SPHEREx uses the position propagated to the frame's own date.
"""

import logging

import astropy.units as u
import pyvo
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time

from . import cache
from .config import settings
from .models import Resolved
from .targets import FEATURED, SIMBAD_IDENTIFIERS, Astrometry, featured_slug

log = logging.getLogger(__name__)

J2000 = Time("J2000")


class ResolveError(Exception):
    pass


def _escape(text: str) -> str:
    return text.replace("'", "''")


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _query_simbad(identifiers: list[str]) -> Astrometry | None:
    quoted = ", ".join(f"'{_escape(i)}'" for i in identifiers)
    adql = (
        "SELECT TOP 1 b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value "
        "FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
        f"WHERE i.id IN ({quoted})"
    )
    errors = []
    for url in settings.simbad_taps:
        try:
            table = pyvo.dal.TAPService(url).run_sync(adql, timeout=20).to_table()
        except Exception as error:
            errors.append(f"{url}: {type(error).__name__}")
            continue
        if len(table) == 0:
            return None
        row = table[0]
        ra, dec = _float(row["ra"]), _float(row["dec"])
        if ra is None or dec is None:
            return None
        return Astrometry(
            name=str(row["main_id"]).strip(),
            ra=ra,
            dec=dec,
            pmra=_float(row["pmra"]) or 0.0,
            pmdec=_float(row["pmdec"]) or 0.0,
            parallax=_float(row["plx_value"]),
            source=f"SIMBAD ({url.split('/')[2]})",
        )
    raise ResolveError("SIMBAD unreachable: " + "; ".join(errors))


def resolve_name(query: str) -> Resolved:
    slug = featured_slug(query)
    cache_name = cache.key("resolve", slug or query.strip().lower())
    cached = cache.read_json("resolve", cache_name)
    if cached:
        return Resolved(**cached)

    identifiers = SIMBAD_IDENTIFIERS.get(slug, []) if slug else [query.strip()]
    found: Astrometry | None = None
    try:
        found = _query_simbad(identifiers)
    except ResolveError as error:
        log.warning("%s", error)
        if not slug:
            raise
    if found is None and slug:
        found = FEATURED[slug]
    if found is None:
        raise ResolveError(f"No SIMBAD object matches '{query}'")

    resolved = Resolved(
        query=query,
        name=FEATURED[slug].name if slug else found.name,
        simbad_id=None if "built-in" in found.source else found.name,
        ra=found.ra,
        dec=found.dec,
        pmra=found.pmra,
        pmdec=found.pmdec,
        parallax=found.parallax,
        epoch="J2000.0",
        source=found.source,
        featured=slug,
    )
    if "built-in" not in resolved.source:
        cache.write_json("resolve", cache_name, resolved.model_dump())
    return resolved


def coordinates(ra: float, dec: float) -> Resolved:
    return Resolved(
        query=f"{ra} {dec}",
        name="Custom position",
        simbad_id=None,
        ra=ra,
        dec=dec,
        pmra=0.0,
        pmdec=0.0,
        parallax=None,
        epoch="as given",
        source="user coordinates",
        featured=None,
    )


def _moving(target: Resolved) -> SkyCoord:
    plx = target.parallax
    distance = Distance(parallax=plx * u.mas) if plx and plx > 0 else 1 * u.kpc
    return SkyCoord(
        ra=target.ra * u.deg,
        dec=target.dec * u.deg,
        pm_ra_cosdec=target.pmra * u.mas / u.yr,
        pm_dec=target.pmdec * u.mas / u.yr,
        distance=distance,
        radial_velocity=0 * u.km / u.s,
        obstime=J2000,
        frame="icrs",
    )


def position_at(target: Resolved, mjd: float) -> SkyCoord:
    if target.pmra == 0 and target.pmdec == 0:
        return SkyCoord(target.ra * u.deg, target.dec * u.deg, frame="icrs")
    moved = _moving(target).apply_space_motion(new_obstime=Time(mjd, format="mjd"))
    return SkyCoord(moved.ra, moved.dec, frame="icrs")
