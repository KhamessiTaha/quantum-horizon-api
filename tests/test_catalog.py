import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from app import catalog

# Invented sources, not real Gaia rows: one still, one moving a full arcsecond a year north.
ROWS = [
    {"source_id": "still-1", "ra": 269.4463, "dec": 4.7682, "pmra": 0.0, "pmdec": 0.0, "parallax": 1.0, "g_mag": 12.0},
    {"source_id": "mover-1", "ra": 269.4463, "dec": 4.7682, "pmra": 0.0, "pmdec": 1000.0, "parallax": 50.0, "g_mag": 9.0},
]


def test_sources_are_propagated_from_gaias_2016_epoch():
    ten_years_on = Time("J2026.0").mjd
    moved = catalog.at_epoch(ROWS, ten_years_on)
    start = SkyCoord([r["ra"] for r in ROWS] * u.deg, [r["dec"] for r in ROWS] * u.deg)
    shifts = start.separation(moved).arcsec
    assert shifts[0] < 0.1
    assert 9.0 < shifts[1] < 11.0
    assert moved[1].dec.deg > start[1].dec.deg


def test_fast_movers_are_recognised():
    assert catalog.moves_more_than(ROWS[1], 0.5)
    assert not catalog.moves_more_than(ROWS[0], 0.5)


def test_the_endpoint_reports_a_gaia_outage_rather_than_guessing(client, monkeypatch):
    def fail(*_args, **_kwargs):
        raise catalog.CatalogError("Gaia archive unavailable: ConnectionError")

    monkeypatch.setattr(catalog, "sources", fail)
    response = client.get("/api/catalog", params={"target": "Barnard's Star"})
    assert response.status_code == 503
    assert "Gaia" in response.json()["detail"]


def test_the_endpoint_places_sources_on_the_grid(client, monkeypatch):
    monkeypatch.setattr(catalog, "sources", lambda *_a, **_k: (ROWS, "Gaia DR3"))
    body = client.get("/api/catalog", params={"target": "Barnard's Star"}).json()
    assert body["provider"] == "Gaia DR3"
    assert body["sources"], "both invented sources sit at the field centre, so both should be drawn"
    for source in body["sources"]:
        for place in source["positions"].values():
            assert -2 <= place[0] <= 40
            assert -2 <= place[1] <= 40
    mover = next(s for s in body["sources"] if s["source_id"] == "mover-1")
    assert mover["pm_total_arcsec_per_year"] == pytest.approx(1.0, abs=0.01)
    assert len(mover["positions"]) >= 2
    first, last = list(mover["positions"].values())[0], list(mover["positions"].values())[-1]
    assert first[1] != last[1], "a moving source must be drawn in a different place in each pass"
