"""Warm the cache for the demo targets, so nothing is fetched live while anyone is watching.

Reading pixels is the slow part: about 100 seconds for 60 frames from the S3 mirror. Once a target is cached, every
view of it, and every measurement, spectrum and light curve, is served from disk in well under a second, and keeps
working if IRSA is down.

Run from the backend folder:

    .venv\\Scripts\\python scripts/prewarm.py
    .venv\\Scripts\\python scripts/prewarm.py --targets barnard rw-aur --detectors D1 D2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import analysis, movers  # noqa: E402
from app.pipeline import build_view  # noqa: E402
from app.resolve import resolve_name  # noqa: E402
from app.targets import FEATURED  # noqa: E402
from app.view import NoCoverage, prepare  # noqa: E402

# The detector each target is worth showing on, from the research phase.
DEFAULT_PLAN: dict[str, list[str]] = {
    "barnard": ["D2", "D1"],
    "eps-indi-b": ["D5"],
    "wise-0855": ["D6"],
    "rw-aur": ["D1"],
    "r-dra": ["D1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_PLAN), choices=list(FEATURED))
    parser.add_argument("--detectors", nargs="*", default=None, help="override the detectors for every target")
    parser.add_argument("--fov", type=float, default=0.06)
    parser.add_argument("--movers", action="store_true", help="also run the mover search, which needs Gaia")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0
    for slug in args.targets:
        try:
            target = resolve_name(FEATURED[slug].name)
        except Exception as error:
            print(f"{slug}: could not resolve: {error}")
            failures += 1
            continue
        for detector in args.detectors or DEFAULT_PLAN.get(slug, ["D2"]):
            started = time.perf_counter()
            label = f"{slug} {detector}"
            try:
                frames = build_view(target, detector, args.fov)
            except NoCoverage as error:
                print(f"{label}: {error}")
                continue
            except Exception as error:
                print(f"{label}: FAILED {type(error).__name__}: {str(error)[:160]}")
                failures += 1
                continue

            view = prepare(target, detector, args.fov)
            analysis.spectrum(view)
            analysis.light_curve(view)
            if args.movers:
                try:
                    movers.search(view)
                except Exception as error:
                    print(f"{label}: mover search skipped: {type(error).__name__}")

            used = sum(1 for f in frames.frames if f.status == "used")
            motion = frames.measurement.motion if frames.measurement else None
            summary = (
                f"{motion.measured_arcsec}″ ± {motion.sigma_arcsec} vs {motion.predicted_arcsec} predicted"
                if motion and motion.measured_arcsec is not None
                else "no motion measured"
            )
            print(
                f"{label}: {used}/{len(frames.frames)} frames, {len(frames.passes)} passes, {summary} "
                f"[{time.perf_counter() - started:.0f}s]"
            )
    print("\nCache is warm." if not failures else f"\n{failures} target(s) failed; rerun them before the demo.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
