"""The mover search, checked on synthetic fields where the truth is known.

The research spike measured this detector over 150 fields: 0.01 false alarms per field, and reliable detection above
about 6 arcseconds per year for a bright source on a one-year baseline. These tests pin the behaviour that mattered:
it finds a planted mover, it is not fooled by a variable star, and it stays quiet on a field with nothing moving.
"""

import numpy as np
import pytest

from app import movers

FIELD = 117
YEARS = [0.0, 0.52, 0.99]
NOISE = 1.0


def psf(shape, x, y, flux, sigma=movers.PSF_SIGMA):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return flux / (2 * np.pi * sigma**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def field(seed: int, mover=None, variable=False):
    """Three passes over a crowded field, optionally with one moving source and one variable star."""
    rng = np.random.default_rng(seed)
    per_snr = NOISE * np.sqrt(4 * np.pi) * movers.PSF_SIGMA
    count = 60
    stars = {
        "x": rng.uniform(6, FIELD - 7, count),
        "y": rng.uniform(6, FIELD - 7, count),
        "flux": np.minimum(6 * (1 - rng.random(count)) ** (-1 / 1.2), 400) * per_snr,
    }
    passes = []
    for year in YEARS:
        image = rng.normal(0, NOISE, (FIELD, FIELD))
        for i in range(count):
            flux = stars["flux"][i]
            if variable and i == 0:
                flux *= {0.0: 1.0, 0.52: 2.4, 0.99: 0.5}[year]
            image += psf(image.shape, stars["x"][i], stars["y"][i], flux)
        if mover is not None:
            speed_px, snr = mover
            image += psf(image.shape, 70.0 + speed_px * year, 30.0 + speed_px * year, snr * per_snr)
        passes.append(image)
    return passes


def run(passes):
    detections = [movers.detect(image) for image in passes]
    return movers.link(detections, YEARS), detections


def test_detect_finds_sources_and_reports_a_position_error():
    image = np.random.default_rng(1).normal(0, NOISE, (40, 40)) + psf((40, 40), 20.0, 15.0, 200.0)
    found = movers.detect(image)
    assert found, "a 200-sigma source must be detected"
    best = max(found, key=lambda s: s["snr"])
    assert best["x"] == pytest.approx(20.0, abs=0.3)
    assert best["y"] == pytest.approx(15.0, abs=0.3)
    assert best["ex"] >= movers.POSITION_FLOOR_PX


def test_a_planted_mover_is_found_with_the_right_speed():
    speed_px = 1.6  # about 10 arcseconds a year, Barnard's Star territory
    tracks, _ = run(field(seed=3, mover=(speed_px, 120)))
    moving = [t for t in tracks if t["significance"] >= movers.MOTION_SIGMA and not t["crowded"]]
    assert moving, "the planted mover should survive the motion threshold"
    best = max(moving, key=lambda t: t["speed"])
    assert best["speed"] == pytest.approx(speed_px * np.sqrt(2), rel=0.25)


def test_a_still_field_produces_no_movers():
    tracks, _ = run(field(seed=5))
    assert not [t for t in tracks if t["significance"] >= movers.MOTION_SIGMA and not t["crowded"]]


def test_a_variable_star_is_not_mistaken_for_a_mover():
    tracks, _ = run(field(seed=7, variable=True))
    assert not [t for t in tracks if t["significance"] >= movers.MOTION_SIGMA and not t["crowded"]]


def test_the_endpoint_refuses_to_search_two_passes(client, monkeypatch):
    from app import view as view_module

    original = view_module.processing.group_passes
    monkeypatch.setattr(view_module.processing, "group_passes", lambda mjds: [list(range(len(mjds)))])
    body = client.get("/api/movers", params={"target": "Barnard's Star"}).json()
    monkeypatch.setattr(view_module.processing, "group_passes", original)
    assert body["candidates"] == []
    assert "at least" in " ".join(body["notes"])
