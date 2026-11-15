r"""Store IRSA MOST's list of SPHEREx exposures for each asteroid that can be blinked, so they open without IRSA.

Run from backend/ (MOST takes a few minutes per body):

    .venv\Scripts\python scripts/index_tracks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.track import BODIES, query_most, store, stored  # noqa: E402


def main() -> int:
    end = Time.now().iso[:10]
    for name, number in BODIES.items():
        if stored(name) is not None and "--refresh" not in sys.argv:
            print(f"{name}: stored")
            continue
        for attempt in range(3):
            try:
                found = query_most(str(number), end)
                store(name, found)
                print(f"{name}: {len(found)} exposures", flush=True)
                break
            except Exception as error:
                print(f"{name}: attempt {attempt + 1} failed ({type(error).__name__}: {str(error)[:120]})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
