r"""Check JPL's orbits against SPHEREx for the brightest asteroids, and write the results for the survey page.

Each asteroid is tracked through its SPHEREx exposures on one detector exactly as the /track page does, and its mean
offset from JPL's predicted position is recorded with its uncertainty and confidence label. Run from backend/:

    .venv\Scripts\python scripts/survey_asteroids.py --count 150 --detector D1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.track import TrackError, track  # noqa: E402

DATA = ROOT.parent / "src" / "data"
EXCLUDED = {"TNO", "CEN"}
OUT = DATA / "asteroid_survey.json"


def candidates(count: int) -> list[dict]:
    raw = json.loads((DATA / "smallbodies.json").read_text(encoding="utf-8"))
    fields = raw["fields"]
    rows = [dict(zip(fields, row)) for row in raw["bodies"]]
    asteroids = [
        r for r in rows
        if r["group"] in {"asteroid", "neo", "famous"} and r["class"] not in EXCLUDED and r.get("designation") and r.get("H") is not None
    ]
    asteroids.sort(key=lambda r: r["H"])
    return asteroids[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--detector", default="D1")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"results": []}
    done = {r["designation"]: r for r in previous["results"] if r["detector"] == args.detector and r["class"] not in EXCLUDED}
    results = list(done.values())

    lock = threading.Lock()

    def survey(index: int, body: dict) -> None:
        started = time.time()
        entry = {
            "name": body["name"],
            "designation": body["designation"],
            "class": body["class"],
            "H": body["H"],
            "diameter_km": body["diameter_km"],
            "detector": args.detector,
        }
        try:
            r = track(body["name"], args.detector, body["designation"], "asteroid")
            m = r["measurement"]
            entry.update(
                status="ok",
                exposures=r["exposures_total"],
                frames=len(r["frames"]),
                detected_frames=sum(1 for f in r["frames"] if f["offset_arcsec"] is not None),
                saturated_frames=sum(1 for f in r["frames"] if f["saturated"]),
                typical_vmag=r["typical_vmag"],
                stack_snr=r["stack_snr"],
                offset_arcsec=round(m["mean_offset_arcsec"], 2) if m["mean_offset_arcsec"] is not None else None,
                sigma_arcsec=round(m["sigma_arcsec"], 2) if m["sigma_arcsec"] is not None else None,
                label=m["label"],
            )
        except TrackError as error:
            entry.update(status="no-data", reason=str(error)[:160])
        except Exception as error:
            entry.update(status="failed", reason=f"{type(error).__name__}: {str(error)[:140]}")
        with lock:
            results.append(entry)
            OUT.write_text(
                json.dumps({"detector": args.detector, "generated": time.strftime("%Y-%m-%d"), "results": results}, indent=1),
                encoding="utf-8",
            )
            print(f"{index:4d} {body['name']:<22} {entry.get('label', entry['status']):<15} "
                  f"{entry.get('offset_arcsec')} ± {entry.get('sigma_arcsec')}  ({time.time() - started:.0f} s)", flush=True)

    todo = [(i, b) for i, b in enumerate(candidates(args.count), start=1) if b["designation"] not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item: survey(*item), todo))
    print(f"wrote {OUT} with {len(results)} asteroids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
