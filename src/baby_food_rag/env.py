"""Read ``.env`` at the repo root, so OPENROUTER_API_KEY need not be exported.

A variable already set in the environment wins over the file. The repo
root is the folder that holds ``sources/manifest.yaml``. ``.env`` is
gitignored. Nothing here prints a value.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from baby_food_rag.manifest import find_manifest


def load_env(root: Path | None = None) -> bool:
    """True when a ``.env`` file was found and read."""
    if root is None:
        try:
            root = find_manifest().parents[1]
        except FileNotFoundError:
            return False
    path = root / ".env"
    if not path.is_file():
        return False
    return load_dotenv(path, override=False)
