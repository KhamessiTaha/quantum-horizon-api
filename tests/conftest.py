import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app import discovery, pixels, resolve
from app.config import settings
from app.main import app
from app.resolve import ResolveError
from tests.synthetic import write_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture()
def env(tmp_path, monkeypatch, dataset):
    records, files = dataset
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    state = {"tap_calls": 0, "tap_down": False, "s3_fail": set()}

    def fake_tap(ra, dec):
        state["tap_calls"] += 1
        if state["tap_down"]:
            raise discovery.DiscoveryError("IRSA TAP unavailable: 503")
        return list(records)

    def fake_s3(record, center, target, half_px):
        if record.plane_id in state["s3_fail"]:
            raise OSError("simulated S3 failure")
        with fits.open(files[record.uri]) as hdul:
            return pixels._extract(hdul, center, target, half_px, "s3")

    def fake_cutout(record, center, target, half_px, pixel_arcsec):
        raise OSError("simulated cutout failure")

    def no_simbad(identifiers):
        raise ResolveError("SIMBAD unreachable (test)")

    monkeypatch.setattr(discovery, "_query_tap", fake_tap)
    monkeypatch.setattr(pixels, "_read_s3", fake_s3)
    monkeypatch.setattr(pixels, "_read_cutout", fake_cutout)
    monkeypatch.setattr(pixels.time, "sleep", lambda s: None)
    monkeypatch.setattr(resolve, "_query_simbad", no_simbad)
    return state


@pytest.fixture()
def client(env):
    return TestClient(app)
