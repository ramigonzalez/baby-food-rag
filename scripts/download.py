#!/usr/bin/env python3
"""Download official infant-feeding PDFs.

Adapted from the catalog download.sh. PDFs go to data/raw/ and are
gitignored. Checksums in sources/manifest.yaml are verified. A failed
URL is reported and the rest continue. The AAP third-party copy is not
fetched.

    python scripts/download.py
    python scripts/download.py --id MS-2019
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baby_food_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
