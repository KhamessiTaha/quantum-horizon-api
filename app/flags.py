"""SPHEREx FLAGS handling.

Bit names come from each file's own MSKN header cards. QR2 files carry no MSKN cards, so the table below is used
instead; it matches the names read from QR3 headers and the published bit list for the Spectrophotometry Tool, where
every flag's value is 2**bit.

Why the four groups matter. A bright star's own pixels carry TRANSIENT, OVERFLOW, SUR_ERROR, SOURCE and BLOOM: the
on-board sample-up-the-ramp logic reports a transient and a possible ramp error wherever a source is bright, and BLOOM
marks the halo around it. Rejecting those pixels deletes the star, which is what happened to every QR3 frame of
Barnard's Star before this split existed. So saturation-family flags are kept and reported as a brightness caveat,
while only genuinely unusable pixels are masked.
"""

import numpy as np

STANDARD_NAMES: dict[int, str] = {
    0: "TRANSIENT",
    1: "OVERFLOW",
    2: "SUR_ERROR",
    4: "PHANTOM",
    5: "REFERENCE",
    6: "NONFUNC",
    7: "DICHROIC",
    9: "MISSING_DATA",
    10: "HOT",
    11: "COLD",
    12: "FULLSAMPLE",
    14: "PHANMISS",
    15: "NONLINEAR",
    17: "PERSIST",
    19: "OUTLIER",
    20: "CROSSTALK",
    21: "SOURCE",
    22: "GHOST",
    23: "GHOST_FPA",
    24: "GHOST_EXT",
    25: "STREAK",
    26: "BLOOM",
    27: "SNOWBALL",
    28: "HALO",
    29: "SATELLITE_HALO",
}

MEANINGS: dict[str, str] = {
    "TRANSIENT": "transient in the ramp, such as a cosmic-ray hit",
    "OVERFLOW": "nearing saturation",
    "SUR_ERROR": "possible error in the ramp fit, common on bright sources",
    "NONFUNC": "dead pixel",
    "DICHROIC": "unusable pixel in the dichroic region",
    "MISSING_DATA": "no data",
    "HOT": "noisy pixel",
    "COLD": "dead pixel",
    "FULLSAMPLE": "full ramp history available",
    "PHANMISS": "phantom correction missing",
    "NONLINEAR": "beyond the nonlinearity correction",
    "PERSIST": "persistent charge from an earlier exposure",
    "OUTLIER": "flagged as an outlier",
    "CROSSTALK": "detector crosstalk",
    "SOURCE": "pixel belongs to a known source",
    "GHOST": "optical ghost of a bright source",
    "GHOST_FPA": "ghost from elsewhere on the focal plane",
    "GHOST_EXT": "ghost from a source outside the field",
    "STREAK": "satellite or debris streak",
    "BLOOM": "blooming around a saturated source",
    "SNOWBALL": "cosmic-ray snowball",
    "HALO": "transient halo",
    "SATELLITE_HALO": "halo from a satellite",
}

# Pixels we refuse to use: no usable signal, or signal we cannot correct.
UNUSABLE = {"NONFUNC", "COLD", "HOT", "MISSING_DATA", "DICHROIC", "PHANMISS", "OUTLIER", "SNOWBALL", "STREAK"}
# Kept, because they sit on real sources, but brightness from them is indicative only.
SATURATION = {"OVERFLOW", "NONLINEAR", "BLOOM", "SUR_ERROR", "PERSIST"}
# Kept, and reported: light that belongs to something else may be in the pixel.
CONTAMINATION = {"GHOST", "GHOST_FPA", "GHOST_EXT", "CROSSTALK", "HALO", "SATELLITE_HALO"}
INFORMATIONAL = {"SOURCE", "FULLSAMPLE", "PHANTOM", "REFERENCE"}
# A transient is a cosmic ray unless the pixel belongs to a known source, where it is the usual bright-source flag.
TRANSIENT = "TRANSIENT"
SOURCE = "SOURCE"


def names_from(header) -> tuple[dict[int, str], bool]:
    """Bit names from the file's MSKN cards, falling back to the standard table. True when the fallback was used."""
    found: dict[int, str] = {}
    for card in header.cards:
        if card.keyword.startswith("MSKN") and card.keyword[4:].isdigit():
            found[int(card.keyword[4:])] = str(card.value).strip()
    if found:
        return found, False
    return dict(STANDARD_NAMES), True


def resolve(table: dict[int, str]) -> dict[int, str]:
    """Replace placeholder names with the standard table, so QR2 and QR3 classify the same way."""
    if not table or all(name.startswith("BIT") for name in table.values()):
        return dict(STANDARD_NAMES)
    return table


def bit_of(table: dict[int, str], name: str) -> int | None:
    for bit, value in table.items():
        if value == name:
            return bit
    return None


def mask_bits(table: dict[int, str], group: set[str]) -> int:
    bits = 0
    for bit, name in table.items():
        if name in group:
            bits |= 1 << bit
    return bits


def bad_pixels(flags: np.ndarray, table: dict[int, str]) -> np.ndarray:
    """Pixels to exclude: unusable flags anywhere, plus transients away from a known source."""
    table = resolve(table)
    bad = (flags & mask_bits(table, UNUSABLE)) != 0
    transient = bit_of(table, TRANSIENT)
    source = bit_of(table, SOURCE)
    if transient is not None:
        hit = (flags & (1 << transient)) != 0
        if source is not None:
            hit &= (flags & (1 << source)) == 0
        bad |= hit
    return bad


def names(mask: int, table: dict[int, str]) -> list[str]:
    table = resolve(table)
    return [table.get(bit, f"BIT{bit}") for bit in range(34) if mask >> bit & 1]


def classify(found: list[str]) -> dict[str, list[str]]:
    return {
        "unusable": [f for f in found if f in UNUSABLE],
        "saturation": [f for f in found if f in SATURATION],
        "contamination": [f for f in found if f in CONTAMINATION],
        "informational": [f for f in found if f in INFORMATIONAL or f == TRANSIENT],
    }


def describe(found: list[str]) -> list[str]:
    return [f"{name}: {MEANINGS[name]}" for name in found if name in MEANINGS]
