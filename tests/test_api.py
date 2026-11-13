import io

import numpy as np
from PIL import Image

from tests.synthetic import FRAMES_PER_PASS, PASS_STARTS


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_resolve_featured_falls_back_to_built_in(client):
    body = client.get("/api/resolve", params={"q": "barnards star"}).json()
    assert body["featured"] == "barnard"
    assert body["source"] == "SIMBAD (built-in copy)"
    assert body["pmdec"] == 10362.4


def test_resolve_unknown_name_is_404(client):
    response = client.get("/api/resolve", params={"q": "Not A Real Star 123"})
    assert response.status_code == 404


def test_frames_builds_three_passes(client):
    response = client.get("/api/frames", params={"target": "Barnard's Star", "det": "D2", "fov": 0.06})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["passes"]) == len(PASS_STARTS)
    assert body["difference_image"]
    assert body["grid"]["width_px"] == body["grid"]["height_px"]
    statuses = [f["status"] for f in body["frames"]]
    assert statuses.count("off-detector") == 1
    assert statuses.count("used") == len(PASS_STARTS) * FRAMES_PER_PASS - 1
    assert body["releases"] == ["qr2", "qr3"]
    assert any("proper motion" in n for n in body["notes"])


def test_predicted_track_moves_north_and_matches_the_star(client):
    body = client.get("/api/frames", params={"target": "Barnard's Star"}).json()
    ys = [p["predicted_px"][1] for p in body["passes"]]
    assert ys[0] > ys[1] > ys[2]
    shift_px = ys[0] - ys[2]
    assert 1.2 < shift_px < 2.2
    for p in body["passes"]:
        png = client.get(p["image"])
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png"
        pixels = np.asarray(Image.open(io.BytesIO(png.content)), dtype=float)
        x, y = p["predicted_px"]
        yy, xx = np.mgrid[0:pixels.shape[0], 0:pixels.shape[1]]
        near = np.hypot(xx - x, yy - y) <= 4
        weights = np.where(near, pixels, 0) ** 2
        cy = float((yy * weights).sum() / weights.sum())
        cx = float((xx * weights).sum() / weights.sum())
        assert abs(cx - x) < 0.6 and abs(cy - y) < 0.6


def test_second_request_is_served_from_cache(client):
    first = client.get("/api/frames", params={"target": "Barnard's Star"}).json()
    second = client.get("/api/frames", params={"target": "Barnard's Star"}).json()
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert second["passes"] == first["passes"]


def test_irsa_outage_uses_stored_image_list(client, env):
    client.get("/api/frames", params={"target": "Barnard's Star"})
    env["tap_down"] = True
    response = client.get("/api/frames", params={"target": "Barnard's Star", "refresh": True})
    assert response.status_code == 200
    assert any("IRSA is unavailable" in n for n in response.json()["notes"])


def test_irsa_outage_without_stored_list_is_503(client, env):
    env["tap_down"] = True
    response = client.get("/api/frames", params={"target": "Barnard's Star"})
    assert response.status_code == 503


def test_failed_reads_are_reported_not_hidden(client, env):
    env["s3_fail"] = {"plane003", "plane010"}
    body = client.get("/api/frames", params={"target": "Barnard's Star"}).json()
    failed = [f for f in body["frames"] if f["status"] == "failed"]
    assert {f["plane_id"] for f in failed} == {"plane003", "plane010"}
    assert any("could not be read" in n for n in body["notes"])


def test_custom_coordinates_have_no_predicted_track(client):
    body = client.get("/api/frames", params={"ra": 269.4463, "dec": 4.7682}).json()
    assert body["target"]["name"] == "Custom position"
    assert all(p["predicted_px"] is None for p in body["passes"])


def test_detector_without_images_is_404(client):
    response = client.get("/api/frames", params={"target": "Barnard's Star", "det": "D6"})
    assert response.status_code == 404


def test_image_names_are_validated(client):
    assert client.get("/api/images/..%2F..%2Fsecret.png").status_code == 404
    assert client.get("/api/images/notavalidname.png").status_code == 404


def test_bad_query_is_400(client):
    assert client.get("/api/frames").status_code == 400
    assert client.get("/api/frames", params={"ra": 400, "dec": 0}).status_code == 422
