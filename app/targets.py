"""Astrometry for the five challenge targets, copied from SIMBAD (values from Gaia DR3 where available).

Used when SIMBAD cannot be reached, so the featured targets always resolve. Positions are ICRS at epoch J2000.0;
proper motions in mas/yr (pmra includes cos dec); parallax in mas.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Astrometry:
    name: str
    ra: float
    dec: float
    pmra: float
    pmdec: float
    parallax: float | None
    source: str


FEATURED: dict[str, Astrometry] = {
    "barnard": Astrometry("Barnard's Star", 269.452077, 4.693365, -801.6, 10362.4, 546.9759, "SIMBAD (built-in copy)"),
    "eps-indi-b": Astrometry("Epsilon Indi B", 331.044147, -56.782844, 3982.0, -2466.8, 270.658, "SIMBAD (built-in copy)"),
    "wise-0855": Astrometry("WISE 0855-0714", 133.818908, -7.247048, -8123.7, 673.2, 439.0, "SIMBAD (built-in copy)"),
    "rw-aur": Astrometry("RW Aurigae", 76.956526, 30.401438, 7.2, -20.3, 15.42, "SIMBAD (built-in copy)"),
    "r-dra": Astrometry("R Draconis", 248.167611, 66.754979, -20.6, 6.1, 1.2509, "SIMBAD (built-in copy)"),
}

ALIASES: dict[str, str] = {
    "barnard": "barnard",
    "barnards star": "barnard",
    "barnard s star": "barnard",
    "gj 699": "barnard",
    "gl 699": "barnard",
    "eps ind b": "eps-indi-b",
    "epsilon indi b": "eps-indi-b",
    "eps indi b": "eps-indi-b",
    "e ind b": "eps-indi-b",
    "wise 0855": "wise-0855",
    "wise 0855 0714": "wise-0855",
    "wise j085510 83 071442 5": "wise-0855",
    "wisea j085510 74 071442 5": "wise-0855",
    "rw aur": "rw-aur",
    "rw aurigae": "rw-aur",
    "v rw aur": "rw-aur",
    "r dra": "r-dra",
    "r draconis": "r-dra",
    "v r dra": "r-dra",
}

SIMBAD_IDENTIFIERS: dict[str, list[str]] = {
    "barnard": ["NAME Barnard's Star", "GJ 699"],
    "eps-indi-b": ["* eps Ind B", "eps Ind B"],
    "wise-0855": ["WISEA J085510.74-071442.5", "WISE J085510.83-071442.5"],
    "rw-aur": ["V* RW Aur"],
    "r-dra": ["V* R Dra"],
}


def normalize(name: str) -> str:
    text = name.lower().replace("*", " ").replace("'", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def featured_slug(name: str) -> str | None:
    norm = normalize(name)
    if norm in FEATURED:
        return norm
    if norm.replace(" ", "-") in FEATURED:
        return norm.replace(" ", "-")
    return ALIASES.get(norm)
