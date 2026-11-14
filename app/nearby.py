"""Named objects around a position, from SIMBAD, so the sky map can say what is there.

Positions are SIMBAD's catalog coordinates (J2000). They label the map; nothing here is measured. The best-known objects
come first, ranked by how often they are cited.
"""

import logging
import re

import pyvo

from . import cache
from .config import settings

log = logging.getLogger(__name__)

MAX_OBJECTS = 40
PREFIXES = re.compile(r"^(NAME|V\*|\*)\s+")


class NearbyError(Exception):
    pass


def _display(main_id: str) -> str:
    return re.sub(r"\s+", " ", PREFIXES.sub("", main_id.strip()))


def named_objects(ra: float, dec: float, radius: float) -> dict:
    ra, dec, radius = round(ra % 360, 3), round(dec, 3), round(radius, 3)
    key = f"{ra:.3f}_{dec:+.3f}_{radius:.3f}"
    stored = cache.read_json("nearby", key)
    if stored:
        return stored

    adql = (
        f"SELECT TOP {MAX_OBJECTS} b.main_id AS name, b.ra AS ra, b.dec AS dec, d.description AS kind, b.nbref AS refs "
        "FROM basic AS b LEFT JOIN otypedef AS d ON d.otype = b.otype "
        f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1 "
        "ORDER BY refs DESC"
    )
    errors = []
    for url in settings.simbad_taps:
        try:
            table = pyvo.dal.TAPService(url).run_sync(adql, timeout=25).to_table()
        except Exception as error:
            errors.append(f"{url.split('/')[2]}: {type(error).__name__}")
            continue
        objects = [
            {
                "name": str(row["name"]).strip(),
                "display": _display(str(row["name"])),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "kind": str(row["kind"]).strip() if row["kind"] is not None and str(row["kind"]) != "--" else "",
                "refs": int(row["refs"]) if str(row["refs"]) != "--" else 0,
            }
            for row in table
        ]
        result = {"ra": ra, "dec": dec, "radius": radius, "objects": objects, "source": f"SIMBAD ({url.split('/')[2]})"}
        cache.write_json("nearby", key, result)
        return result
    raise NearbyError("SIMBAD unreachable: " + "; ".join(errors))
