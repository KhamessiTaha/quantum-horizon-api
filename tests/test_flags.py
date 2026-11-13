import numpy as np
from astropy.io import fits

from app import flags


def bit(name: str) -> int:
    return 1 << flags.bit_of(flags.STANDARD_NAMES, name)


def test_qr2_headers_without_mskn_cards_fall_back_to_the_standard_table():
    table, fallback = flags.names_from(fits.Header())
    assert fallback
    assert table[2] == "SUR_ERROR"
    assert flags.resolve({0: "BIT0", 21: "BIT21"})[21] == "SOURCE"


def test_a_bright_star_keeps_its_pixels():
    """The pixels of a bright source carry TRANSIENT, OVERFLOW, SUR_ERROR, SOURCE and BLOOM together.

    Rejecting those deletes the star, which is what happened to every QR3 frame of Barnard's Star before this rule.
    """
    star = bit("TRANSIENT") | bit("OVERFLOW") | bit("SUR_ERROR") | bit("SOURCE") | bit("BLOOM")
    assert not flags.bad_pixels(np.array([[star]]), flags.STANDARD_NAMES)[0, 0]


def test_a_cosmic_ray_away_from_a_source_is_rejected():
    assert flags.bad_pixels(np.array([[bit("TRANSIENT")]]), flags.STANDARD_NAMES)[0, 0]
    assert flags.bad_pixels(np.array([[bit("SNOWBALL")]]), flags.STANDARD_NAMES)[0, 0]


def test_dead_and_missing_pixels_are_rejected_and_ghosts_are_kept():
    for name in ("NONFUNC", "COLD", "HOT", "MISSING_DATA", "STREAK", "OUTLIER"):
        assert flags.bad_pixels(np.array([[bit(name)]]), flags.STANDARD_NAMES)[0, 0], name
    for name in ("GHOST", "CROSSTALK", "HALO", "PERSIST", "NONLINEAR"):
        assert not flags.bad_pixels(np.array([[bit(name)]]), flags.STANDARD_NAMES)[0, 0], name


def test_classification_splits_the_groups():
    found = flags.names(bit("SOURCE") | bit("BLOOM") | bit("GHOST") | bit("HOT"), flags.STANDARD_NAMES)
    grouped = flags.classify(found)
    assert grouped["unusable"] == ["HOT"]
    assert grouped["saturation"] == ["BLOOM"]
    assert grouped["contamination"] == ["GHOST"]
    assert "SOURCE" in grouped["informational"]
