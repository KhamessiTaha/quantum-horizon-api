from pathlib import Path

from app.track import parse_most, stored

SAMPLE = Path(__file__).parent / "data" / "most_vesta_sample.txt"


def test_parse_most_reads_irsa_rows():
    sightings = parse_most(SAMPLE.read_text(encoding="utf-8"))
    assert len(sightings) == 4
    first = sightings[0]
    assert first.detector in {"D1", "D2", "D3", "D4", "D5", "D6"}
    assert 0 <= first.ra < 360 and -90 <= first.dec <= 90
    assert 60700 < first.mjd < 61500
    assert first.uri.endswith(".fits") and "/spherex/" in first.uri


def test_stored_asteroids_ship_with_the_api():
    vesta = stored("Vesta")
    assert vesta and len(vesta) > 100
    assert {s.detector for s in vesta} >= {"D1", "D2"}
